from __future__ import print_function, division
import json
import os
from utils import calculate_ndvi_by_crop, clean_crop_times, estimate_planting_harvest_periods_by_crop, mask_to_text, mask_to_text_ndvi
from pastis24.data_transforms import Normalize
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.utils.data
import pickle
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

with open("../datalake/label_names_en.json", "r") as json_file:
    LABEL_NAMES_EN = json.load(json_file)

def get_distr_dataloader(paths_file, root_dir, rank, world_size, transform=None, batch_size=32, num_workers=4,
                         shuffle=True, return_paths=False):
    """
    return a distributed dataloader
    """
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=True, sampler=sampler)
    return dataloader


def get_dataloader(paths_file, root_dir, transform=None, batch_size=32, num_workers=4, shuffle=True,
                   return_paths=False, my_collate=None):
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=my_collate)
    return dataloader


class SatImDataset(Dataset):
    """Satellite Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None, multilabel=False, return_paths=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if type(csv_file) == str:
            self.data_paths = pd.read_csv(csv_file, header=None)
        elif type(csv_file) in [list, tuple]:
            self.data_paths = pd.concat([pd.read_csv(csv_file_, header=None) for csv_file_ in csv_file], axis=0).reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.multilabel = multilabel
        self.return_paths = return_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.data_paths.iloc[idx, 0])

        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')

        # Crear transform sin Normalize
        if self.transform:
            transform_wo_norm = transforms.Compose(
                [t for t in self.transform.transforms if not isinstance(t, Normalize)]
            )
            sample_wo_norm = transform_wo_norm(sample.copy())
        else:
            sample_wo_norm = sample.copy()

        # Usar sample sin normalización para calcular NDVI
        label_mask = sample_wo_norm['labels']
        inputs = sample_wo_norm['inputs']
        cleaned_ndvi_by_crop = clean_crop_times(inputs, label_mask[:, :, 0])
        planting_harvest = estimate_planting_harvest_periods_by_crop(cleaned_ndvi_by_crop)
        text = mask_to_text_ndvi(label_mask[:, :, 0], planting_harvest)

        # Aplicar transformaciones completas (con Normalize) a sample original
        if self.transform:
            sample = self.transform(sample)

        if self.return_paths:
            return sample, text, img_name

        return sample, text


    def read(self, idx, abs=False):
        """
        read single dataset sample corresponding to idx (index number) without any data transform applied
        """
        if type(idx) == int:
            img_name = os.path.join(self.root_dir,
                                    self.data_paths.iloc[idx, 0])
        if type(idx) == str:
            if abs:
                img_name = idx
            else:
                img_name = os.path.join(self.root_dir, idx)
        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')
        return sample
    
    
def my_collate(batch):
    "Filter out sample where mask is zero everywhere"
    idx = [b['unk_masks'].sum(dim=(0, 1, 2)) != 0 for b in batch]
    batch = [b for i, b in enumerate(batch) if idx[i]]
    return torch.utils.data.dataloader.default_collate(batch)
