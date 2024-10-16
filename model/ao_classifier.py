from typing import Any

from pytorch_lightning import LightningModule
from torch import nn
from torchvision import models
import torch
from torchmetrics import classification, MetricCollection
from clearml import Task, Logger


class AOClassifier(LightningModule):
    def __init__(self, resnet_depth: int = 34, n_classes: int = 8,
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
        try:
            self.model = {
                18: models.resnet18,
                34: models.resnet34,
                50: models.resnet50
            }[resnet_depth]
        except KeyError:
            raise NotImplementedError(f"ResNet-{resnet_depth} is not implemented.")
        self.model = self.model(weights='DEFAULT')
        # replace first conv depending on input config
        self.model.conv1 = nn.Conv2d(n_input_channel, 64, 7, 2, 3, bias=False)
        self.latent_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self.classifier = nn.Linear(self.latent_dim, n_classes)

        # loss
        self.bce = nn.BCEWithLogitsLoss()

        # metrics
        self.train_metrics = MetricCollection({
            "acc": classification.MultilabelAccuracy(num_labels=n_classes, average=None),
            "f1": classification.MultilabelF1Score(num_labels=n_classes, average=None)
        }, postfix='/train')
        self.val_metrics = self.train_metrics.clone(postfix='/val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
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

    def forward(self, batch):
        use_image, use_frac_loc, use_bin_seg, use_mult_seg = self.input_config
        x = torch.cat([
            batch['image'] if use_image else torch.empty(0, device=self.device),
            batch['fracture_heatmap'] if use_frac_loc else torch.empty(0, device=self.device),
            batch['segmentation'].any(1) if use_bin_seg else torch.empty(0, device=self.device),
            batch['segmentation'] if use_mult_seg else torch.empty(0, device=self.device)
        ])

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

    def training_step(self, batch):
        return self.step_with_monitoring(batch, "train")

    def on_train_epoch_end(self) -> None:
        epoch_values = self.train_metrics.compute()
        class_labels = self.trainer.datamodule.train_dataset.CLASS_LABELS
        for name, value in epoch_values.items():
            Logger.current_logger().report_histogram(name, 'train', value.cpu().numpy(), self.current_epoch,
                                                     xaxis='class', yaxis=name, xlabels=class_labels)

        self.train_metrics.reset()

    def validation_step(self, batch):
        return self.step_with_monitoring(batch, "val")

    def on_validation_epoch_end(self) -> None:
        epoch_values = self.val_metrics.compute()
        class_labels = self.trainer.datamodule.train_dataset.CLASS_LABELS
        for name, value in epoch_values.items():
            Logger.current_logger().report_histogram(name, 'val', value.cpu().numpy(), self.current_epoch,
                                                     xaxis='class', yaxis=name, xlabels=class_labels)
        self.val_metrics.reset()
