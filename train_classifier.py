from shutil import rmtree

from clearml import Task
from pytorch_lightning.cli import LightningCLI
from model.ao_classifier import AOClassifier
from dataset.grazpedwri_dataset import GrazPedWriDataModule

task = Task.init(project_name="AO Classification", auto_resource_monitoring=False, reuse_last_task_id=False)

# training routine
cli = LightningCLI(model_class=AOClassifier, datamodule_class=GrazPedWriDataModule)

# housekeeping
trainer = cli.trainer
Task.current_task().upload_artifact("best.ckpt", trainer.checkpoint_callback.best_model_path, wait_on_upload=True)
Task.current_task().close()
rmtree(trainer.logger.log_dir)