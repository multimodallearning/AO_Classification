from pathlib import Path

import torch
from clearml import Task
from tqdm import tqdm

import clearml_ids
from dataset.grazpedwri_dataset import GrazPedWriDataModule
from model.ao_classifier import AOClassifier

save_dir = Path("evaluation/predictions")

data = GrazPedWriDataModule()
data.setup("test")

# save ground truth
ground_truth = {}
for batch in data.test_dataloader():
    for file_name, y in zip(batch['file_name'], batch['y']):
        ground_truth[file_name] = y
torch.save(ground_truth, save_dir / "ground_truth.pt")

experiment_dict = vars(clearml_ids)
# remove python variables
experiment_dict = {k: v for k, v in experiment_dict.items() if not k.startswith("__")}

for experiment, clearml_id in experiment_dict.items():
    assert isinstance(clearml_id, str), f"clearml_ids.{experiment} must be a string"
    predictions = {}

    ckpt_path = Task.get_task(clearml_id).artifacts['best.ckpt'].get()
    model = AOClassifier.load_from_checkpoint(ckpt_path, strict=False)
    model.eval()

    with torch.inference_mode():
        for batch in tqdm(data.test_dataloader(), desc=f"Save predictions for {experiment}"):
            y_hat = model(batch).sigmoid()
            predictions.update({file_name: y for file_name, y in zip(batch['file_name'], y_hat)})

    torch.save(predictions, save_dir / f"{experiment}_{clearml_id}.pt")
