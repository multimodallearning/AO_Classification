# script iterating ver all predictions and calculating the metrics

from pathlib import Path

import pandas as pd
import torch
from torchmetrics import classification, MetricCollection

from dataset.grazpedwri_dataset import GrazPedWriDataset

mode = ['end2end', 'lin_eval'][0]

metrics_kwargs = {'num_labels': GrazPedWriDataset.N_CLASSES, 'average': None}
metrics = MetricCollection({
    "Accuracy": classification.MultilabelAccuracy(**metrics_kwargs),
    "F1": classification.MultilabelF1Score(**metrics_kwargs),
    "Precision": classification.MultilabelPrecision(**metrics_kwargs),
    "Recall": classification.MultilabelRecall(**metrics_kwargs),
    "AUROC": classification.MultilabelAUROC(**metrics_kwargs)
})
pred_dir = Path('evaluation/predictions')
gt = torch.load(pred_dir / 'ground_truth.pt')

mean_df = pd.DataFrame(columns=['Experiment', 'Accuracy', 'F1', 'Precision', 'Recall', 'AUROC'])
experiment_df = pd.DataFrame(columns=['Experiment', 'Accuracy', 'F1', 'Precision', 'Recall', 'AUROC', 'AO_Class'])
for experiment in pred_dir.iterdir():
    is_line_eval = experiment.stem.startswith('LE')
    match_mode = (mode == 'lin_eval' and is_line_eval) or (mode == 'end2end' and not is_line_eval)
    contains_bin_seg = 'bin_seg' in experiment.stem
    if experiment.stem == 'ground_truth' or experiment.is_dir() or not match_mode or contains_bin_seg:
        continue

    pred = torch.load(experiment)
    y = []
    y_hat = []
    for file_stem in gt.keys():
        y.append(gt[file_stem])
        y_hat.append(pred[file_stem])
    y = torch.stack(y).int()
    y_hat = torch.stack(y_hat)

    performance = metrics(y_hat, y)
    mean_df = pd.concat([mean_df, pd.DataFrame({
        'Experiment': experiment.stem.rsplit('_', 1)[0],
        'Accuracy': performance['Accuracy'].mean().item(),
        'F1': performance['F1'].mean().item(),
        'Precision': performance['Precision'].mean().item(),
        'Recall': performance['Recall'].mean().item(),
        'AUROC': performance['AUROC'].mean().item()
    }, index=[0]), ], ignore_index=True)

    experiment_df = pd.concat([experiment_df, pd.DataFrame({
        'Experiment': experiment.stem.rsplit('_', 1)[0],
        'Accuracy': performance['Accuracy'].tolist(),
        'F1': performance['F1'].tolist(),
        'Precision': performance['Precision'].tolist(),
        'Recall': performance['Recall'].tolist(),
        'AUROC': performance['AUROC'].tolist(),
        'AO_Class': GrazPedWriDataset.CLASS_LABELS
    }), ], ignore_index=True)

experiment_df.set_index(['Experiment', 'AO_Class'], inplace=True)
experiment_df.sort_index(inplace=True)
print(experiment_df.to_string())
print()

mean_df.set_index('Experiment', inplace=True)
mean_df.sort_values('AUROC', ascending=False, inplace=True)
print(mean_df.to_markdown())
