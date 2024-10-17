from typing import Any

from pytorch_lightning import LightningModule
from torch import nn
from torchvision import models
import torch
from torchmetrics import classification, MetricCollection
from clearml import Task, Logger


class AOClassifier(LightningModule):
    def __init__(self, n_classes: int = 8,
                 use_image: bool = True, use_frac_loc: bool = False, use_bin_seg: bool = False,
                 use_mult_seg: bool = False):
        if (use_bin_seg or use_mult_seg) and (use_bin_seg == use_mult_seg):
            raise AssertionError("Binary and multilabel segmentation should be used exclusively.")
        super().__init__()
        # input config
        self.input_config = (use_image, use_frac_loc, use_bin_seg, use_mult_seg)
        assert any(self.input_config), "At least one input type must be used."
        n_input_channel = sum(self.input_config)
        if use_mult_seg:
            n_input_channel += 16  # 17 bones, but 1 is already added by use_mult_seg

        # model
        self.model = models.mobilenet_v3_small(weights='DEFAULT')
        # replace first conv depending on input config
        self.model.features[0][0] = nn.Conv2d(n_input_channel, 16, 3, 2, 1, bias=False)
        self.latent_dim = self.model.classifier[0].in_features
        self.model.classifier = nn.Identity()

        self.classifier = nn.Linear(self.latent_dim, n_classes)

        # loss
        self.bce = nn.BCEWithLogitsLoss()

        # metrics
        self.train_metrics = MetricCollection({
            "acc": classification.MultilabelAccuracy(num_labels=n_classes, average=None),
            "f1": classification.MultilabelF1Score(num_labels=n_classes, average=None),
            "prec": classification.MultilabelPrecision(num_labels=n_classes, average=None),
            "rec": classification.MultilabelRecall(num_labels=n_classes, average=None)
        }, postfix='/train')
        self.val_metrics = self.train_metrics.clone(postfix='/val')

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs,
        #                                                        eta_min=self.hparams.lr / 100)
        return optimizer

    def on_fit_start(self) -> None:
        self.bce.pos_weight = self.trainer.datamodule.train_dataset.BCE_POS_WEIGHT.to(self.device)

        task_id = []
        use_image, use_frac_loc, use_bin_seg, use_mult_seg = self.input_config
        if use_image:
            task_id.append('image')
        if use_frac_loc:
            task_id.append('frac_loc')
        if use_bin_seg:
            task_id.append('bin_seg')
        if use_mult_seg:
            task_id.append('mult_seg')

        if Task.current_task() is not None:
            Task.current_task().set_name(f'{"_".join(task_id)}')
            Task.current_task().set_tags([f'fold {self.trainer.datamodule.fold}'])

    def forward(self, batch):
        use_image, use_frac_loc, use_bin_seg, use_mult_seg = self.input_config
        x = torch.cat([
            batch['image'] if use_image else torch.empty(0, device=self.device),
            batch['fracture_heatmap'] if use_frac_loc else torch.empty(0, device=self.device),
            batch['segmentation'].any(1, keepdim=True) if use_bin_seg else torch.empty(0, device=self.device),
            batch['segmentation'] if use_mult_seg else torch.empty(0, device=self.device)
        ], dim=1)

        features = self.model(x)
        y_hat = self.classifier(features)
        return y_hat

    def step_with_monitoring(self, batch, mode):
        y_hat = self(batch)
        y = batch['y']
        loss = self.bce(y_hat, y)

        with torch.no_grad():
            # metrics
            metrics = self.train_metrics if mode == 'train' else self.val_metrics
            batch_values = metrics(y_hat, y)

            # logging
            logg_kwargs = {"on_step": False, "on_epoch": True, "batch_size": len(y)}
            self.log(f"bce/{mode}", loss, **logg_kwargs)
            for name, value in batch_values.items():
                self.log(name, value.mean(), **logg_kwargs)

        return loss

    def report_histogram(self, metric_collection:MetricCollection, mode:str):
        if Logger.current_logger() is None:
            return

        epoch_values = metric_collection.compute()
        class_labels = self.trainer.datamodule.train_dataset.CLASS_LABELS
        for name, value in epoch_values.items():
            name = name.split('/')[0]
            Logger.current_logger().report_histogram(name, mode, value.cpu().numpy(), self.current_epoch,
                                                     xaxis='class', yaxis=name, xlabels=class_labels)

        metric_collection.reset()


    def training_step(self, batch):
        return self.step_with_monitoring(batch, "train")

    def on_train_epoch_end(self) -> None:
        self.report_histogram(self.train_metrics, 'train')

    def validation_step(self, batch):
        return self.step_with_monitoring(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self.report_histogram(self.val_metrics, 'val')