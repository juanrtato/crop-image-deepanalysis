import os
import pandas as pd
import numpy as np
import pickle

import os
import pandas as pd
import numpy as np
import pickle

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

