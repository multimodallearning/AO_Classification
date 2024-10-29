from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torchmetrics import classification

from dataset.grazpedwri_dataset import GrazPedWriDataset

models2plot = ['image', 'image_frac_loc', 'image_frac_loc_mult_seg_clip']

pred_dir = Path('evaluation/predictions')
gt = torch.load(pred_dir / 'ground_truth.pt')

path_dict = {k.stem.rsplit('_', 1)[0]: k for k in pred_dir.iterdir() if not k.is_dir()}
pred_dict = dict()
for experiment in models2plot:
    pred = torch.load(path_dict[experiment])
    y = []
    y_hat = []
    for file_stem in gt.keys():
        y.append(gt[file_stem])
        y_hat.append(pred[file_stem])
    y = torch.stack(y).int()
    y_hat = torch.stack(y_hat)
    pred_dict[experiment] = y_hat


roc = classification.MultilabelROC(num_labels=len(models2plot))
for c in range(GrazPedWriDataset.N_CLASSES):
    cat_y_hat = torch.stack([pred_dict[e][:, c] for e in models2plot], dim=1)

    roc.update(cat_y_hat, y[:, c].unsqueeze(1).expand(-1, len(models2plot)))
    fig, axs = roc.plot(score=True, labels=models2plot)
    axs.set_title('')
    plt.legend(fontsize='large')

    fig.savefig(f'/home/ron/Documents/Konferenzen/BVM 2025/ROCs/roc_{GrazPedWriDataset.CLASS_LABELS[c].replace('/', '_')}.pdf',
                bbox_inches='tight', pad_inches=0)

    roc.reset()
plt.show()


