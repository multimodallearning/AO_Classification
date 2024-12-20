# script to preprocess the data and save it to a h5 file to speed up the training process
from pathlib import Path

import h5py
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from dataset.grazpedwri_dataset import GrazPedWriDataset
from dataset.heatmap_extractor import HeatmapExtractor

save_dir = Path('data/preprocessed_data.h5')

img_path = Path('data/img_only_front_all_left')
yolo_label_path = Path('data/yolo_labels/ground_truth')
yolo_prediction_path = Path('data/yolo_labels/predictions')

df_meta = pd.read_csv('data/dataset_cv_splits.csv', index_col='filestem')
h5_dataset = h5py.File(save_dir, 'x')
h5_dataset.attrs['n_classes'] = GrazPedWriDataset.N_CLASSES
h5_dataset.attrs['class_labels'] = GrazPedWriDataset.CLASS_LABELS
# load data
h5_saved_seg = h5py.File('data/segmentations_all.h5', 'r')['segmentation_mask']
heatmap_extractor = HeatmapExtractor(resolution_HW=GrazPedWriDataset.RESCALE_HW, class2extract=3)  # 3 = fracture
for file_name in tqdm(df_meta.index, unit='img', desc=f'Preprocessing data'):
    need2flip = df_meta.loc[file_name, 'laterality'] == 'R'

    # image
    img = Image.open(img_path.joinpath(file_name).with_suffix('.png')).convert('L')
    img = img.resize(GrazPedWriDataset.RESCALE_HW[::-1], Image.BILINEAR)
    img = to_tensor(img)

    # segmentation
    seg = h5_saved_seg[file_name][:]

    # fracture heatmap
    heatmap_gt = heatmap_extractor.extract_heatmap(yolo_label_path.joinpath(file_name).with_suffix('.txt'))
    try:
        heatmap_pred = heatmap_extractor.extract_heatmap(yolo_prediction_path.joinpath(file_name).with_suffix('.txt'))
    except FileNotFoundError:
        print(f'No prediction found for {file_name}. This is normal, if no predictable object was found, but should be'
              f'checked, because we will create a zero heatmap for this case leading into a silent error.')
        heatmap_pred = torch.zeros_like(heatmap_gt)
    if need2flip:
        heatmap_gt = heatmap_gt.flip(1)
        heatmap_pred = heatmap_pred.flip(1)

    # classification ground truth
    class_label: str = df_meta.loc[file_name, 'ao_classification']
    class_label: list[str] = class_label.split(';')
    y = torch.zeros(GrazPedWriDataset.N_CLASSES)
    for c in class_label:
        c = c.strip()
        if c not in GrazPedWriDataset.CLASS_IDX:
            continue
        else:
            y[GrazPedWriDataset.CLASS_IDX[c]] = 1
    assert y.sum() > 0, f'No valid class found for {file_name} with {class_label}'

    h5_dataset.create_group(file_name)
    h5_dataset[file_name]['image'] = img
    h5_dataset[file_name]['segmentation'] = seg
    h5_dataset[file_name]['fracture_heatmap_ground_truth'] = heatmap_gt
    h5_dataset[file_name]['fracture_heatmap_yolo_prediction'] = heatmap_pred
    h5_dataset[file_name]['y'] = y

h5_dataset.close()
