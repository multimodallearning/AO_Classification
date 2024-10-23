from shutil import rmtree

from clearml import Task
from pytorch_lightning.cli import LightningCLI
from model.linear_evaluation import LinearEvaluation
from dataset.grazpedwri_dataset import GrazPedWriDataModule

task = Task.init(project_name="AO Classification/linear evaluation", auto_resource_monitoring=False, reuse_last_task_id=False)

# training routine
cli = LightningCLI(model_class=LinearEvaluation, datamodule_class=GrazPedWriDataModule)

# housekeeping
trainer = cli.trainer
Task.current_task().upload_artifact("best.ckpt", trainer.checkpoint_callback.best_model_path, wait_on_upload=True)
Task.current_task().close()
rmtree(trainer.logger.log_dir)
