import torch
from torchmetrics import classification, MetricCollection
from dataset.grazpedwri_dataset import GrazPedWriDataset
import pandas as pd
from pathlib import Path

mode = ['end2end', 'lin_eval'][1]

metrics_kwargs = {'num_labels': GrazPedWriDataset.N_CLASSES, 'average': None}
metrics = MetricCollection({
    "Acc@1": classification.MultilabelAccuracy(**metrics_kwargs),
    "F1": classification.MultilabelF1Score(**metrics_kwargs),
    "Precision": classification.MultilabelPrecision(**metrics_kwargs),
    "Recall": classification.MultilabelRecall(**metrics_kwargs),
    "AUROC": classification.MultilabelAUROC(**metrics_kwargs)
})
pred_dir = Path('evaluation/predictions')
gt = torch.load(pred_dir / 'ground_truth.pt')

mean_df = pd.DataFrame(columns=['Experiment', 'Acc@1', 'F1', 'Precision', 'Recall', 'AUROC'])
experiment_df = pd.DataFrame(columns=['Experiment', 'Acc@1', 'F1', 'Precision', 'Recall', 'AUROC', 'AO_Class'])
for experiment in pred_dir.iterdir():
    is_line_eval = experiment.stem.startswith('LE')
    match_mode = (mode == 'lin_eval' and is_line_eval) or (mode == 'end2end' and not is_line_eval)
    if experiment.stem == 'ground_truth' or experiment.is_dir() or not match_mode:
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
        'Acc@1': performance['Acc@1'].mean().item(),
        'F1': performance['F1'].mean().item(),
        'Precision': performance['Precision'].mean().item(),
        'Recall': performance['Recall'].mean().item(),
        'AUROC': performance['AUROC'].mean().item()
    }, index=[0]), ], ignore_index=True)

    experiment_df = pd.concat([experiment_df, pd.DataFrame({
        'Experiment': experiment.stem.rsplit('_', 1)[0],
        'Acc@1': performance['Acc@1'].tolist(),
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
