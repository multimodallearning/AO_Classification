import torch
from torchmetrics.functional import classification
from dataset.grazpedwri_dataset import GrazPedWriDataset
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
# supress warnings
import warnings

warnings.filterwarnings("ignore")

candidate = "image_frac_loc_bin_seg_clip"
test_direction = ['greater', 'two-sided'][0]
significance_level = 0.05

pred_dir = Path('evaluation/predictions')
available_experiments = [experiment for experiment in pred_dir.iterdir() if
                         experiment.stem != 'ground_truth' and not experiment.is_dir()]

metric = lambda y_hat, y: classification.multilabel_auroc(y_hat, y, num_labels=GrazPedWriDataset.N_CLASSES,
                                                          average=None)
gt = torch.load(pred_dir / 'ground_truth.pt')
filelist = list(gt.keys())
gt = torch.stack([gt[file_stem] for file_stem in filelist]).int()

candidate_path = [experiment for experiment in available_experiments if experiment.name.rsplit('_', 1)[0] == candidate][0]
y_pred_canditate = torch.load(candidate_path)
y_pred_canditate = torch.stack([y_pred_canditate[file_stem] for file_stem in filelist])
auroc_canditate = metric(y_pred_canditate, gt)
print(f'Candidate: {candidate_path.name.rsplit('_', 1)[0]} with AUROC: {auroc_canditate.mean().item()}')
print(f'Test direction: {test_direction}')

df = pd.DataFrame(columns=['Challenger', 'AUROC', 'statistic', 'p-value', f'significant at {significance_level}'])
for challenger in available_experiments:
    if challenger == candidate_path:
        continue
    y_pred_challenger = torch.load(challenger)
    y_pred_challenger = torch.stack([y_pred_challenger[file_stem] for file_stem in filelist])

    auroc_challenger = metric(y_pred_challenger, gt)

    test_result = wilcoxon(auroc_canditate, auroc_challenger, alternative='greater')
    df = pd.concat([df, pd.DataFrame({
        'Challenger': challenger.stem.rsplit('_', 1)[0],
        'AUROC': auroc_challenger.mean().item(),
        'statistic': test_result.statistic,
        'p-value': test_result.pvalue,
        f'significant at {significance_level}': test_result.pvalue < significance_level
    }, index=[0]), ], ignore_index=True)

df.set_index('Challenger', inplace=True)
df.sort_index(inplace=True)
df.sort_values('p-value', ascending=False, inplace=True)
print(df.to_string())
