import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from mappings import *


METADATA_PATH = '../datalake/PASTIS24/metadata.geojson'
with open(METADATA_PATH, 'r') as f:
    metadata_json = json.load(f)

data = []
for feature in metadata_json["features"]:
    properties = feature["properties"]
    data.append(properties)

METADATA_DF = pd.DataFrame(data)


class CustomSequenceDataset:
    def __init__(self, csv_path, root_dir, max_T=60, batch_size=32):
        self.paths_df = pd.read_csv(csv_path, header=None)  
        self.root_dir = root_dir
        self.max_T = max_T
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.paths_df))

    def __len__(self):
        return len(self.paths_df)

    def __getitem__(self, idx):
        relative_path = self.paths_df.iloc[idx, 0]
        full_path = os.path.join(self.root_dir, relative_path)

        with open(full_path, 'rb') as f:
            data = pickle.load(f)

        images = data['img']
        labels = data['labels']
        doys = data['doy']  

        T = images.shape[0]
        C, H, W = images.shape[1:]

        # Reorganize the dims (T, C, H, W) to (T, H, W, C)
        images = np.transpose(images, (0, 2, 3, 1))  # (T, C, H, W) to (T, H, W, C)

        if T < self.max_T:
            padding = np.zeros((self.max_T - T, H, W, C))
            images = np.concatenate([images, padding], axis=0)
            doys = np.pad(doys, (0, self.max_T - T), mode='constant', constant_values=0)
            mask = np.concatenate([np.ones(T), np.zeros(self.max_T - T)])
        else:
            images = images[:self.max_T]
            doys = doys[:self.max_T]
            mask = np.ones(self.max_T)

        return {
            'inputs': images,           
            'labels': np.array(labels, dtype=int),   
            'seq_lengths': T,
            'unk_masks': mask,            
            'doy': np.array(doys)
        }

    def get_batches(self):
        np.random.shuffle(self.indexes)  # Shuffle indexes for randomness
        for start_idx in range(0, len(self), self.batch_size):
            batch_indexes = self.indexes[start_idx:start_idx + self.batch_size]
            batch_data = [self[i] for i in batch_indexes]
            yield batch_data


def get_date_from_id(patch_id, index, metadata_df):
    row = metadata_df[metadata_df['ID_PATCH'] == int(patch_id)]
    if not row.empty:
        date_dict = row.iloc[0]['dates-S2']
        date = date_dict.get(str(index), None)
        date = str(datetime.strptime(str(date), "%Y%m%d").strftime("%Y/%m/%d")) if date else None
        return date
    else:
        return None

def plot_seg_mask(mask_array, colormap, class_labels=None):
    """
    Plots a segmentation mask with a custom colormap.
    
    Args:
        mask_array (np.ndarray): 2D array representing the segmentation mask.
        colormap (list): List of RGB tuples for the colormap.
        class_labels (dict, optional): Dictionary mapping class indices to labels.
    
    """
    custom_cmap = mcolors.ListedColormap(colormap)
    bounds = np.arange(len(colormap) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.imshow(mask_array, cmap=custom_cmap, norm=norm)
    cbar = fig.colorbar(cax, ax=ax, ticks=np.arange(len(colormap)))
    cbar.set_ticks(np.arange(len(colormap)))
    cbar.set_ticklabels(list(class_labels.values()))
    ax.set_title("Segmentation Mask")
    ax.axis('off')  
    plt.show()


def plot_sample_rgb(inputs, labels, patch_id, sample_idx=-1):
    """
    Plots a sample RGB image from the dataset.

    Args:
        inputs (np.ndarray): Input data array of shape (T, H, W, C).
        labels (np.ndarray): Labels array of shape (T, H, W).
        patch_id (str): Identifier for the patch.
        sample_idx (int): Index of the sample to plot. If -1, a random sample is selected.
    
    """
    if hasattr(inputs, 'numpy'):
        inputs = inputs.numpy()
        labels = labels.numpy()

    rgb_indices = [2, 1, 0]
    sample = inputs[sample_idx]
    sample_labels = labels[sample_idx]

    print(f"Sample shape: {sample.shape}")
    print(f"Sample labels shape: {sample_labels.shape}")

    fig, axes = plt.subplots(6, 10, figsize=(20, 12))
    fig.suptitle(f"Temporal evolution of sample #{sample_idx}", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i >= sample.shape[0]:
            ax.axis('off')
            continue

        img = sample[i]  # (24, 24, 11)
        rgb = img[:, :, rgb_indices]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-5)

        ax.imshow(rgb)
        ax.set_title(f'T{i} - {get_date_from_id(patch_id, i, METADATA_DF)}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_mean_band_profile(inputs, sample_idx):
    """
    Plots the mean value of each band over time for a single sample.

    Args:
        inputs: tensor of shape [N, T, H, W, B]
        sample_idx: index of the sample to analyze
    """
    sample = inputs[sample_idx]  # shape: [T, H, W, B]
    num_times = sample.shape[0]

    df = pd.DataFrame(columns=['time', 'value', 'band'])

    for t_index in range(num_times):
        # Calculate the mean pixel value for each band at time t_index
        mean_pixel_values = sample[t_index, :, :, :10].mean(dim=(0, 1))  # Mean over height and width
        
        for band_index, mean_value in enumerate(mean_pixel_values):
            df = pd.concat([df, pd.DataFrame([{
                'time': t_index,
                'value': mean_value.item(),  # Convert tensor to Python float
                'band': f"B{band_index + 1}"
            }])], ignore_index=True)

    # Plot the figure
    plt.figure(figsize=(10, 6))
    for band in sorted(df['band'].unique()):
        band_df = df[df['band'] == band]
        plt.plot(band_df['time'], band_df['value'], label=band)

    plt.xlabel("Time")
    plt.ylabel("Mean Value")
    plt.title(f"Mean Pixel Band - Sample {sample_idx}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def mask_to_text(mask_array: np.ndarray, label_names: dict, language: str = "en") -> str:
    """
    Convert a segmentation mask to a text representation.
    
    Args:
        mask_array (np.ndarray): Segmentation mask with class labels.
        label_names (dict): Dictionary mapping class indices to labels.
        language (str): Language for the output text ("es" for Spanish, "en" for English).
    
    Returns:
        str: Text representation of the segmentation mask.
    """
    if language == "es":
        initial_msg = INITIAL_MSGS_ES
        end_msg = END_MSGS_ES
        no_crop_msg = NO_CROP_MSGS_ES
        ext_msg = EXTENTION_MSGS_ES
    elif language == "en":
        initial_msg = INITIAL_MSGS_EN
        end_msg = END_MSGS_EN
        no_crop_msg = NO_CROP_MSGS_EN
        ext_msg = EXTENTION_MSGS_EN
    else:
        raise ValueError("Language not supported. Use 'es' for Spanish or 'en' for English.")
        
    initial_msg = initial_msg[np.random.randint(0, len(initial_msg))]
    area_per_class = get_area_per_class(mask_array, label_names)
    class_names = list(area_per_class.keys())
    if len(class_names) == 0 or all(
        class_name in ["Etiqueta vacía", "Fondo", "Background", "Void label"]
        for class_name in class_names
    ):
        return no_crop_msg[np.random.randint(0, len(no_crop_msg))]

    total_area = sum(area_per_class.values())
    for class_name, area in area_per_class.items():
        if class_name not in ["Etiqueta vacía", "Fondo", "Background", "Void label"]:
            initial_msg += f"{class_name.lower()} {ext_msg[np.random.randint(0, len(ext_msg))]} {area} m2, "
        #else:
        # TODO: que hacer con el fondo y etiqueta vacia?

    initial_msg += f"{end_msg[np.random.randint(0, len(end_msg))]}{total_area} m2."
    text_representation = initial_msg + "\n"

    return text_representation


def get_area_per_class(seg_mask: np.array, label_names: dict, area_px_m2: int = 100) -> dict:
    """
    Get the area (in m2) of each class in the segmentation mask.
    Args:
        seg_mask (np.ndarray): Segmentation mask with class labels.
        classes (dict): Dictionary mapping class labels to their respective names.
        area_px_m2 (float): Area of one pixel in square meters.
        label_names (dict): Dictionary mapping class indices to labels.
    
    Returns:
        dict: Dictionary with class labels as keys and their respective areas in square meters.
    """
    classes, counts = np.unique(seg_mask, return_counts=True)
    if label_names is not None:
        areas_per_class = {
            label_names.get(str(int(crop_class))): int(count * area_px_m2)
            for crop_class, count in zip(classes, counts)
        }
        return areas_per_class

    return {
            str(int(crop_class)): int(count * area_px_m2)
            for crop_class, count in zip(classes, counts)
        }