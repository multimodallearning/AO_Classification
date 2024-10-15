import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from kornia.enhance import Normalize
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


class GrazPedWriDataset(Dataset):
    # calculated over training split
    IMG_MEAN = 0.3505533917353781
    IMG_STD = 0.22763733675869177

    RESCALE_HW = (384, 224)

    CLASS_LABEL = ['23-M/2.1', '23-M/3.1', '23r-E/2.1', '23r-M/2.1', '23r-M/3.1', '23u-E/7', '23u-M/2.1', 'none']
    CLASS_IDX = {k: v for v, k in enumerate(CLASS_LABEL)}
    N_CLASSES = len(CLASS_LABEL)

    def __init__(self, mode: str, fold: int = 0, number_training_samples: int | str = 'all'):
        super().__init__()
        # load data meta and other information
        self.df_meta = pd.read_csv('data/dataset_cv_splits.csv', index_col='filestem')
        # init ground truth parser considering the data split
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

        # load img into memory
        img_path = Path('data/img_only_front_all_left')
        self.data = dict()
        for file_name in tqdm(self.available_file_names, unit='img', desc=f'Loading data for {mode}'):
            # image
            img = Image.open(img_path.joinpath(file_name).with_suffix('.png')).convert('L')
            img = img.resize(self.RESCALE_HW[::-1], Image.BILINEAR)
            img = to_tensor(img)

            # classification ground truth
            class_label: str = self.df_meta.loc[file_name, 'ao_classification']
            class_label: list[str] = class_label.split(';')
            y = torch.zeros(self.N_CLASSES)
            for c in class_label:
                c = c.strip()
                if c not in self.CLASS_IDX:
                    continue
                else:
                    y[self.CLASS_IDX[c]] = 1
            assert y.sum() > 0, f'No valid class found for {file_name} with {class_label}'

            self.data[file_name] = {
                'file_name': file_name,
                'image': img,
                'y': y

            }
            break

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
    def __init__(self, fold: int = 0, batch_size: int = 32, number_training_samples: int | str = 'all'):
        super().__init__()
        self.n_train = number_training_samples
        self.fold = fold
        self.dl_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
        self.normalize = Normalize(mean=GrazPedWriDataset.IMG_MEAN, std=GrazPedWriDataset.IMG_STD)

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = GrazPedWriDataset('train', self.fold, self.n_train)
            self.val_dataset = GrazPedWriDataset('val', self.fold)
        if stage == 'test' or stage is None:
            self.test_dataset = GrazPedWriDataset('val', self.fold)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, drop_last=True, **self.dl_kwargs)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, **self.dl_kwargs)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, **self.dl_kwargs)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        batch['image'] = self.normalize(batch['image'])
        return batch


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    dataset = GrazPedWriDataset('val', fold=0)
    data = dataset[0]
    print(data['image'].shape)
    print(data['y'])
    plt.figure(data['file_name'])
    plt.imshow(data['image'].squeeze().numpy(), cmap='gray')
    plt.title(dataset.CLASS_LABEL[data['y'].argmax()])
    plt.show()
