from pathlib import Path

import h5py
import torch
from clearml import Task
from kornia.enhance import Normalize
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from dataset.grazpedwri_dataset import GrazPedWriDataset
from model.clip import LightningCLIP

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

clearml_id = 'f91758de8b3b47718225b6f82b728608'
save_dir = Path('data/clip_embed.h5')
normalize = Normalize(mean=GrazPedWriDataset.IMG_MEAN, std=GrazPedWriDataset.IMG_STD).to(device)

ckpt_path = Task.get_task(clearml_id).artifacts['best.ckpt'].get()
clip = LightningCLIP.load_from_checkpoint(ckpt_path).to(device)
clip.eval()

ds = ConcatDataset([GrazPedWriDataset('train'), GrazPedWriDataset('val')])
dl = DataLoader(ds, batch_size=16)

h5_dataset = h5py.File(save_dir, 'x')
h5_dataset.attrs['CLIP_clearml_id'] = clearml_id
with torch.inference_mode():
    for batch in tqdm(dl, desc='Generating CLIP embeddings'):
        batch['image'] = normalize(batch['image'].to(device))
        img_emb, txt_emb = clip(batch)

        for file_name, img, txt in zip(batch['file_name'], img_emb, txt_emb):
            img, txt = img.cpu(), txt.cpu()

            h5_dataset.create_group(file_name)
            h5_dataset[file_name]['image'] = img
            h5_dataset[file_name]['text'] = txt

h5_dataset.close()
