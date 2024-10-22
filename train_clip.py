from shutil import rmtree

from clearml import Task
from pytorch_lightning.cli import LightningCLI
from model.clip import LightningCLIP
from dataset.grazpedwri_dataset import GrazPedWriDataModule
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

task = Task.init(project_name="AO Classification/CLIP", auto_resource_monitoring=False, reuse_last_task_id=False)

# training routine
cli = LightningCLI(model_class=LightningCLIP, datamodule_class=GrazPedWriDataModule)

# housekeeping
trainer = cli.trainer
Task.current_task().upload_artifact("best.ckpt", trainer.checkpoint_callback.best_model_path, wait_on_upload=True)
Task.current_task().close()
rmtree(trainer.logger.log_dir)
