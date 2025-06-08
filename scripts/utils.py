import os
import random
import re
import torch
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn.functional as F
import dateparser

from mappings import *


METADATA_PATH = '../datalake/PASTIS24/metadata.geojson'
with open(METADATA_PATH, 'r') as f:
    metadata_json = json.load(f)

with open("../datalake/label_names_en.json", "r") as json_file:
    LABEL_NAMES_EN = json.load(json_file)


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


def get_date_from_id(patch_id, index, metadata_df, humanize=False):
    """
    Get the date corresponding to a specific patch ID and index.
    Args:
        patch_id (str): Identifier for the patch.
        index (int): Index of the date to retrieve.
        metadata_df (pd.DataFrame): DataFrame containing metadata information.
        humanize (bool): If True, return a human-readable date format.
    
    Returns:
        str: Date in the format YYYY/MM/DD or None if not found.
    """
    row = metadata_df[metadata_df['ID_PATCH'] == int(patch_id)]
    if not row.empty:
        date_dict = row.iloc[0]['dates-S2']
        date = date_dict.get(str(index), None)
        date = str(datetime.strptime(str(date), "%Y%m%d").strftime("%Y/%m/%d")) if date else None
        if humanize:
            date = humanize_date(date)
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
    if sample_idx == -1:
        sample = inputs
    else:
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


def mask_to_text(mask_array: np.ndarray, label_names: dict = LABEL_NAMES_EN, language: str = "en") -> str:
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
        print("Language not supported yet, using English.")
    initial_msgs = INITIAL_MSGS_EN 
    end_msgs = END_MSGS_EN
    no_crop_msgs = NO_CROP_MSGS_EN
    ext_msgs = EXTENTION_MSGS_EN
        
    area_per_class = get_area_per_class(mask_array, label_names)
    valid_classes = [
        c for c in area_per_class.keys()
        if c not in ["Etiqueta vacía", "Fondo", "Background", "Void label"]
    ]

    if not valid_classes:
        return no_crop_msgs[np.random.randint(0, len(no_crop_msgs))]

    initial_msg_template = initial_msgs[np.random.randint(0, len(initial_msgs))]
    text_representation = initial_msg_template.format(crop=valid_classes[0])

    area = area_per_class[valid_classes[0]]
    ext = ext_msgs[np.random.randint(0, len(ext_msgs))]
    text_representation += f"{ext} {area} m2"

    for class_name in valid_classes[1:]:
        area = area_per_class[class_name]
        ext = ext_msgs[np.random.randint(0, len(ext_msgs))]
        text_representation += f", {class_name} {ext} {area} m2"

    end_msg_selected = end_msgs[np.random.randint(0, len(end_msgs))]
    total_area_info = random.choices([True, False], weights=[0.65, 0.35])[0]
    if total_area_info:
        aprox_values = [
            "approximately",
            "approx.",
            "around",
            "about",
        ]
        total_area = sum(area_per_class.values())
        total_area_msg = f" ({random.choice(aprox_values)} {total_area} m2)"
        end_msg_selected += f"{total_area_msg}."
    else:
        end_msg_selected += "."
    text_representation += f" {end_msg_selected}\n"
    return text_representation


def format_date_phrase(date_str: str) -> str:
    """
    Format a date using the correct preposition based on the date string.
    Args:
        date_str (str): Date string.
    
    Returns:
        str: Formatted date string with appropriate preposition.

    """
    if random.choices([True, False], weights=[0.5, 0.5])[0]:
        return f"{random.choice(['around', 'about', 'by'])} {date_str}"
    else:
        if bool(re.search(r'\b\d{1,2}, \d{4}\b', date_str)):
            return f"on {date_str}"
        else:
            return f"in {date_str}"

    
def mask_to_text_ndvi(mask_array: np.ndarray, planting_harvest: dict, labels: dict = LABEL_NAMES_EN) -> str:
    """
    Convert a segmentation mask and NDVI values to a detailed text description.

    Args:
        mask_array (np.ndarray): Segmentation mask with class labels.
        ndvi_per_crop (dict): Dictionary mapping class indices to NDVI tensors or arrays.
        labels (dict): Mapping of class indices (as str) to crop names.

    Returns:
        str: Textual description including crop area and NDVI health analysis.
    """

    fist_text = mask_to_text(mask_array)
    transition_connector = random.choice(TRANSITION_CONNECTORS_EN)
    ndvi_analysis = []

    for class_idx_str, times_list in planting_harvest.items():
        crop_name = labels.get(class_idx_str, None)
        if not times_list:
            continue
        known_periods = [(p, h) for (p, h) in times_list if p != "unknown"]
        unknown_planting = [(p, h) for (p, h) in times_list if p == "unknown"]
        for _, harvest in unknown_planting:
            no_planting_text = random.choice(NO_PLANTING_MSGS_EN).format(crop=crop_name)
            harvest_text = random.choice(HARVEST_WITHOUT_PLANTING_EN).format(crop=crop_name, harvest_time=harvest)
            ndvi_analysis.append(f"{no_planting_text} {harvest_text}")
        
        if len(known_periods) == 1:
            plant, harvest = known_periods[0]
            msg = random.choice(PLANT_HARVEST_COMBINED_EN).format(crop=crop_name, plant_time=plant, harvest_time=harvest)
            ndvi_analysis.append(msg)
        
        elif len(known_periods) > 1:
            plant_list = [p for (p, _) in known_periods]
            harvest_list = [h for (_, h) in known_periods]
            plant_str = " and ".join(plant_list)
            harvest_str = " and ".join(harvest_list)
            msg = random.choice(MULTI_PLANT_HARVEST_COMBINED_EN).format(
                crop=crop_name, plant_times=plant_str, harvest_times=harvest_str
            )
            ndvi_analysis.append(msg)
        
        
    if not ndvi_analysis:
        return fist_text
    else:
        text_parts = [ndvi_analysis[0]] + [f"{random.choice(OTHER_CONNECTOR_ALTERNATIVES)} {sentence}" for sentence in ndvi_analysis[1:]]
        return fist_text + "\n" + transition_connector + " " + " ".join(text_parts)


def calculate_ndvi(inputs):
    """
    Function to calculate NDVI from the input tensor.
    The input tensor is expected to have the shape [T, H, W, B],
    where T is the number of time steps, H is the height, W is the width,
    and B is the number of bands.
    The NDVI is calculated using the formula:
    NDVI = (NIR - Red) / (NIR + Red)
    where NIR is the value of the NIR band and Red is the value of the Red band.
    The NIR band is assumed to be at index 6 and the Red band at index 7.
    The function returns a tensor with the NDVI values for each time step.
    Args:
        inputs (torch.Tensor): Input tensor of shape [T, H, W, B].
    Returns:
        torch.Tensor: Tensor with the NDVI values for each time step.
    """
    sample = inputs  # shape: [T, H, W, B]
    num_times = sample.shape[0]

    ndvi_values = []

    for t_index in range(num_times):
        nir_band = sample[t_index, :, :, 7] / 10000.0  # B8 → NIR
        red_band = sample[t_index, :, :, 3] / 10000.0  # B4 → Red

        ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-6)
        ndvi_values.append(ndvi)

    return torch.stack(ndvi_values)  # [T, H, W] # Tensor with the NDVI values for each time step


def moving_average(ndvi_tensor, window_size=4):
    """
    Aplica una media móvil simple al tensor 1D de NDVI.
    
    Args:
        ndvi_tensor (torch.Tensor): tensor 1D de NDVI.
        window_size (int): tamaño de la ventana (debe ser impar para centrado).
    
    Returns:
        torch.Tensor: NDVI suavizado.
    """
    padding = window_size // 2
    # reshape para conv1d: (batch=1, canales=1, secuencia)
    ndvi_reshaped = ndvi_tensor.unsqueeze(0).unsqueeze(0)
    kernel = torch.ones(1, 1, window_size) / window_size
    smoothed = F.conv1d(ndvi_reshaped, kernel, padding=padding)
    return smoothed.squeeze()



def calculate_ndvi_by_crop(inputs, sample_labels, label_dict=LABEL_NAMES_EN):
    """
    Calculate the mean NDVI for each crop class in the sample.

    Args:
        inputs (torch.Tensor): Input tensor [N, T, H, W, B].
        sample_labels (torch.Tensor): Mask of segmentation [N, H, W, 1].
        label_dict (dict): Dictionary with class IDs and names.

    Returns:
        dict: {class_id_str: mean_ndvi} for each crop class.
    """
    ndvi_tensor = calculate_ndvi(inputs)  # [T, H, W]
    label_mask = sample_labels
    
    ndvi_by_crop = {}
    for class_id_str, class_name in label_dict.items():
        class_id = int(class_id_str)
        if class_name in ['Background', 'Void label']:
            continue  # ignorar fondo y etiquetas vacías
        
        mask = label_mask == class_id  # [H, W]
        if mask.sum() == 0:
            continue  # clase no presente

        ndvi_per_t = []
        for t in range(ndvi_tensor.shape[0]):
            ndvi_t = ndvi_tensor[t]  # [H, W]
            ndvi_masked = ndvi_t[mask]
            ndvi_nonzero = ndvi_masked[ndvi_masked != 0]
            ndvi_per_t.append(ndvi_nonzero.mean())

        ndvi_by_crop[class_id_str] = torch.stack(ndvi_per_t)  # [T]
    
    #ndvi_by_crop = {k: moving_average(v) for k, v in ndvi_by_crop.items()}

    return ndvi_by_crop


def detect_clouds(inputs, blue_idx=0, swir1_idx=8, swir2_idx=9,
                  blue_thresh=0.2, swir_thresh=0.15):
    """
    Detect clouds in the input tensor using a simple thresholding method.
    This function checks if the blue, SWIR1, and SWIR2 bands exceed certain thresholds
    to determine if clouds are present at each time step.

    Args:
        inputs (torch.Tensor): Tensor [num_samples, T, H, W, B]

    Returns:
        np.ndarray: Boolean array indicating cloud presence at each time step.
    """
    sample = inputs# [T, H, W, B]
    T = sample.shape[0]
    ndvi = calculate_ndvi(inputs)  # [T]
    cloud_mask_by_time = np.zeros(T, dtype=bool)

    for t in range(T):
        blue = sample[t, :, :, blue_idx] / 10000.0
        swir1 = sample[t, :, :, swir1_idx] / 10000.0
        swir2 = sample[t, :, :, swir2_idx] / 10000.0

        cloud_pixel_mask = (blue > blue_thresh) & (swir1 > swir_thresh) & (swir2 > swir_thresh)

        if cloud_pixel_mask.any() or (ndvi[t].mean() < 0.1):
            cloud_mask_by_time[t] = True

    return cloud_mask_by_time


def merge_periods(periods):
    """
    Merge overlapping or contiguous periods in a list of tuples.
    Args:
        periods (list of tuples): List of tuples where each tuple is (start_idx, end_idx).
                                  'unknown' can be used as a placeholder for missing values.
    Returns:
        list of tuples: Merged list of periods.
    """
    if not periods:
        return []
    merged = []
    # Ordenar por start_idx, ignorando 'unknown' que va primero
    periods = sorted(periods, key=lambda x: (float('-inf') if x[0] == "unknown" else x[0]))
    current_start, current_end = periods[0]
    for start, end in periods[1:]:
        # Solo fusionar si ambos índices son números (no "unknown") y si se tocan o solapan
        if current_end == start or (isinstance(current_end, int) and isinstance(start, int) and start <= current_end):
            # Fusionar extendiendo el fin
            if isinstance(end, int) and (not isinstance(current_end, int) or end > current_end):
                current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def estimate_planting_harvest_periods_by_crop(
        ndvi_by_crop, format_dates, patch_id, delta_thresh_start=0.05, delta_thresh_end=0.05, min_duration=5
    ):
    """
    Estimate planting and harvest periods for each crop based on NDVI values.
    This function analyzes the NDVI time series for each crop and detects periods of significant increase (planting)
    and decrease (harvest) in NDVI values. It returns the periods with the highest mean NDVI for each crop.

    Args:
        ndvi_by_crop (dict): {crop_name: NDVI tensor [T]} (can contain NaN values).
        format_dates (bool): If True, the times will be formatted as dates.
        patch_id (str): Identifier for the patch, used for date formatting.
        delta_thresh_start (float): Delta threshold for significant increase to consider planting.
        delta_thresh_end (float): Delta threshold for significant decrease to consider harvest.
        min_duration (int): Minimum duration (in time steps) for a valid planting/harvest period.

    Returns:
        dict: {crop_name: [(start_idx, end_idx)]} List of tuples representing planting and harvest periods.
    """
    crop_periods = {}

    for crop_name, ndvi_tensor in ndvi_by_crop.items():
        ndvi = ndvi_tensor.cpu().numpy()
        mean_ndvi = np.nanmean(ndvi)
        if mean_ndvi <= 0.35:
            crop_periods[crop_name] = []
            continue
        valid_mask = ~np.isnan(ndvi)
        valid_indices = np.where(valid_mask)[0]
        valid_values = ndvi[valid_mask]

        if len(valid_values) < 3:
            crop_periods[crop_name] = []
            continue

        deltas = np.diff(valid_values)
        periods = []
        dict_periods = {}

        # Detectar posible cosecha al inicio sin siembra visible
        #print(f"First ndvi: {ndvi[0]}")
        if deltas[0] < -delta_thresh_end:
            if ndvi[0] >= 0.6:
                periods.append(("unknown", valid_indices[1]))
                dict_periods[("unknown", valid_indices[1])] = ndvi[0]

        i = 0
        while i < len(deltas):
            if deltas[i] > delta_thresh_start:
                start_idx = valid_indices[i]
                for j in range(i + min_duration, len(deltas)):
                    if deltas[j] < -delta_thresh_end:
                        end_idx = valid_indices[j + 1]
                        mean_ndvi_period = np.nanmean(ndvi[start_idx:end_idx])
                        dict_periods[(start_idx, end_idx)] = mean_ndvi_period
                        periods.append((start_idx, end_idx))
                        i = j + 1
                        #break
                else:
                    i += 1
            else:
                i += 1

        best_periods_by_start = {}
        #print(dict_periods)
        for (start, end), mean_val in dict_periods.items():
            if (start not in best_periods_by_start) or (mean_val > best_periods_by_start[start][1]):
                best_periods_by_start[start] = ((start, end), mean_val)

        # Extraer solo las tuplas (start,end)
        best_periods = [v[0] for v in best_periods_by_start.values()]
        best_periods = merge_periods(best_periods)
        crop_periods[crop_name] = best_periods

    if format_dates:
        for crop_name, periods in crop_periods.items():
            formatted_periods = []
            for start, end in periods:
                start_date = get_date_from_id(patch_id, start, METADATA_DF, humanize=True)
                end_date = get_date_from_id(patch_id, end, METADATA_DF, humanize=True)
                if 'by an unknown point in time' in start_date:
                    continue
                if start_date and end_date:
                    formatted_periods.append((start_date, end_date))
                else:
                    formatted_periods.append((start, end))
            crop_periods[crop_name] = formatted_periods
    else:
        for crop_name, periods in crop_periods.items():
            formatted_periods = []
            for start, end in periods:
                if start == 'unknown':
                    continue
                formatted_periods.append((f"on time T{start}", f"on time T{end}"))
            crop_periods[crop_name] = formatted_periods
    return crop_periods


def humanize_date(date: str) -> str:
    """
    Convert a date string in the format "YYYY-MM-DD" to a human-readable format.
    Args:
        date (str): Date string in the format "YYYY-MM-DD".
    Returns:
        str: Human-readable date string in the format "principios/mediados/finales de mes de año".
    """
    try: 
        date = dateparser.parse(date, settings={'DATE_ORDER': 'YMD'})
    except:
        date = None
    if not date:
        return "by an unknown point in time"
    human_date_version = random.choices([True, False], weights=[0.65, 0.35])[0]
    month = date.strftime("%B")
    year = date.year
    day = date.day
    if human_date_version:
        early_options = EARLY_DATE
        mid_options = MID_DATE
        late_options = LATE_DATE
        if day <= 10:
            phrase = random.choice(early_options).format(month=month, year=year)
            
        elif day <= 20:
            phrase = random.choice(mid_options).format(month=month, year=year)
        else:
            phrase = random.choice(late_options).format(month=month, year=year)
        return format_date_phrase(phrase)
    else:
        return format_date_phrase(f"{month} {day}, {year}")


def clean_crop_times(inputs, sample_labels):
    """
    Clean the NDVI data for each crop by removing invalid values and applying cloud detection.
    
    Args:
        inputs (torch.Tensor): Tensor [num_samples, T, H, W, B]
        sample_idx (int): Index of the sample to clean.
        sample_idx (int): Sample index to clean.

    Returns:
        dict: Dictionary with cleaned NDVI tensors:
        {
            'crop_name': cleaned NDVI tensor with invalid values set to NaN
            
        }
    """
    ndvi_by_crop = calculate_ndvi_by_crop(inputs, sample_labels)
    cleaned_data = {}

    for crop_name, ndvi_series in ndvi_by_crop.items():
        cloud_mask = detect_clouds(inputs)  # [T]
        ndvi_values = ndvi_series.cpu().numpy()
        
        valid_ndvi_mask = (ndvi_values >= 0.21)
        valid_mask = valid_ndvi_mask & ~cloud_mask  # NDVI válido y sin nubes

        cleaned_ndvi = ndvi_values.copy()
        cleaned_ndvi[~valid_mask] = np.nan
        cleaned_data[crop_name] = torch.tensor(cleaned_ndvi)

    return cleaned_data



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