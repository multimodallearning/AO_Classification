import logging
from typing import Any

import h5py
import pandas as pd
import torch
from kornia.enhance import Normalize
from kornia.augmentation import RandomAffine, AugmentationSequential
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from tqdm import tqdm


class GrazPedWriDataset(Dataset):
    IMG_MEAN = 0.3505533917353781
    IMG_STD = 0.22763733675869177

    RESCALE_HW = (384, 224)

    CLASS_LABELS = ['23-M/2.1', '23-M/3.1', '23r-E/2.1', '23r-M/2.1', '23r-M/3.1', '23u-E/7', '23u-M/2.1', 'none']
    CLASS_IDX = {k: v for v, k in enumerate(CLASS_LABELS)}
    N_CLASSES = len(CLASS_LABELS)
    BCE_POS_WEIGHT = torch.tensor(
        [16.601983, 14.02660218, 16.18810512, 5.89240155, 3.82040341, 6.73304294, 9.7037037, 3.11217737])

    def __init__(self, mode: str, fold: int = 1, number_training_samples: int | str = 'all',
                 use_yolo_predictions: bool = False):
        super().__init__()
        # load data meta and other information
        self.df_meta = pd.read_csv('data/dataset_cv_splits.csv', index_col='filestem')
        self.df_meta['report'] = self.df_meta['report'].apply(lambda x: 'Kein Befund verfügbar.' if pd.isna(x) else x)

        # init ground truth parser considering the data split
        assert fold <= self.df_meta['fold'].max(), f'max fold index is {self.df_meta["fold"].max()}'
        if mode == 'train':
            self.df_meta = self.df_meta[self.df_meta['fold'] != fold]
        elif mode == 'val':
            self.df_meta = self.df_meta[self.df_meta['fold'] == fold]
        else:
            raise ValueError(f'Unknown mode: {mode}')
        self.available_file_names = self.df_meta.index.tolist()

        # get subset of training samples
        if mode == 'train' and number_training_samples != 'all':
            raise NotImplementedError('number_training_samples is not implemented for GrazPedWriDataset')
        elif mode != 'train' and number_training_samples != 'all':
            logging.warning(f'number_training_samples is not used for mode {mode}')

        # load data into memory
        h5_dataset = h5py.File('data/preprocessed_data.h5', 'r')
        self.data = dict()
        for file_name in tqdm(self.available_file_names, unit='img', desc=f'Loading data for {mode}'):
            img = torch.from_numpy(h5_dataset[file_name]['image'][:])
            seg = torch.from_numpy(h5_dataset[file_name]['segmentation'][:])
            if use_yolo_predictions:
                heatmap = torch.from_numpy(h5_dataset[file_name]['fracture_heatmap_yolo_prediction'][:])
            else:  # use ground truth
                heatmap = torch.from_numpy(h5_dataset[file_name]['fracture_heatmap_ground_truth'][:])
            heatmap = heatmap.unsqueeze(0)  # add channel dimension
            y = torch.from_numpy(h5_dataset[file_name]['y'][:])
            report = self.df_meta.loc[file_name, 'report']

            self.data[file_name] = {
                'file_name': file_name,
                'image': img,
                'segmentation': seg,
                'fracture_heatmap': heatmap,
                'y': y,
                'report': report
            }

    def __len__(self):
        return len(self.available_file_names)

    def __getitem__(self, index):
        """
        get item by index
        :param index: index of item
        :return: dict with keys ['image', 'mask', 'file_name']
        """
        file_name = self.available_file_names[index]
        data_dict = self.data[file_name]

        return data_dict


class GrazPedWriDataModule(LightningDataModule):
    def __init__(self, fold: int = 1, batch_size: int = 64, number_training_samples: int | str = 'all',
                 affine_params_rot_trans_scale: tuple = (30, 0.1, 0.15)):
        super().__init__()
        self.n_train = number_training_samples
        self.fold = fold
        self.dl_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
        self.normalize = Normalize(mean=GrazPedWriDataset.IMG_MEAN, std=GrazPedWriDataset.IMG_STD)

        # data augmentation
        rotate, translate, scale = affine_params_rot_trans_scale
        self.data_aug = AugmentationSequential(
            RandomAffine(degrees=rotate, translate=(translate,) * 2, scale=(1 - scale, 1 + scale), p=1),
            data_keys=['image', 'mask', 'image'])

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = GrazPedWriDataset('train', self.fold, self.n_train)
            self.val_dataset = GrazPedWriDataset('val', self.fold)
        if stage == 'test' or stage is None:
            self.test_dataset = GrazPedWriDataset('val', self.fold, use_yolo_predictions=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, drop_last=True, **self.dl_kwargs)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, **self.dl_kwargs)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, **self.dl_kwargs)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # data augmentation
        if self.trainer.training:
            aug_batch = self.data_aug(batch['image'], batch['segmentation'], batch['fracture_heatmap'])
            batch['image'], batch['segmentation'], batch['fracture_heatmap'] = aug_batch

        batch['image'] = self.normalize(batch['image'])
        return batch


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    dataset = GrazPedWriDataset('val', fold=1, use_yolo_predictions=True)
    data = dataset[0]
    print(data['image'].shape)
    print(data['y'])
    fig, axs = plt.subplots(1, 3, num=data['file_name'])
    axs[0].imshow(data['image'].squeeze().numpy(), cmap='gray')
    axs[0].set_title(dataset.CLASS_LABELS[data['y'].argmax()])

    axs[1].imshow(data['image'].squeeze().numpy(), cmap='gray')
    axs[1].imshow(data['segmentation'].float().argmax(0), alpha=data['segmentation'].any(0).float() * .8, cmap='tab20',
                  interpolation='nearest')

    axs[2].imshow(data['image'].squeeze().numpy(), cmap='gray')
    axs[2].imshow(data['fracture_heatmap'].squeeze(0), cmap='hot', alpha=.8)

    plt.show()
