import torch
from monai.networks.blocks import ResidualUnit
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch import optim
from torchmetrics import MetricCollection
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError
from torchvision import models
from torchvision.utils import make_grid


class ResidualDecoder(nn.Module):
    def __init__(self, num_input_channel: int = 512, num_output_channel: int = 1, num_decode_layer: int = 5):
        super().__init__()
        self.decoder = nn.ModuleList()
        for i in range(num_decode_layer):
            self.decoder.append(
                ResidualUnit(
                    spatial_dims=2,
                    in_channels=num_input_channel,
                    out_channels=num_input_channel // 2,
                    norm='BATCH',
                    act='leakyReLU',
                    bias=False
                )
            )
            self.decoder.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            num_input_channel //= 2
        # add last conv
        self.decoder.append(nn.Conv2d(num_input_channel, num_output_channel, kernel_size=1, stride=1, padding=0))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x = self.decoder(x)
        return x


class LightningAutoEncoder(LightningModule):
    def __init__(self, resnet_depth: int = 18, num_decoder_layers: int = 5, use_ssim: bool = False):
        super().__init__()
        self.plotting_batch = {'train': torch.zeros(1, 1, 64, 64), 'val': torch.zeros(1, 1, 64, 64)}
        self.use_ssim = use_ssim

        # encoder
        try:
            self.img_encoder = {
                18: models.resnet18,
                34: models.resnet34,
                50: models.resnet50
            }[resnet_depth]
        except KeyError:
            raise NotImplementedError(f"ResNet-{resnet_depth} is not implemented.")
        self.img_encoder = self.img_encoder(weights='DEFAULT')
        self.latent_dim = self.img_encoder.fc.in_features
        self.img_encoder.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.img_encoder.avgpool = nn.Identity()
        self.img_encoder.fc = nn.Identity()
        self.decoder = ResidualDecoder(num_input_channel=self.latent_dim, num_decode_layer=num_decoder_layers)

        self.train_metrics = MetricCollection({
            "L1": MeanAbsoluteError(),
            "SSIM": StructuralSimilarityIndexMeasure()})
        self.val_metrics = self.train_metrics.clone()

        self.save_hyperparameters(ignore=self.plotting_batch)

    def forward(self, x):
        spatial_dim = torch.tensor(x.size()[-2:], device=x.device) // 2 ** 5
        x = self.img_encoder(x)
        # view back due to flatten in the model
        x = x.view(x.shape[0], self.latent_dim, spatial_dim[0], spatial_dim[1])
        x = self.decoder(x)
        return x

    def step_with_monitoring(self, mode, batch, metrics):
        x = batch["image"]
        x_hat = self(x)

        # calculate loss
        batch_metrics = metrics(x_hat, x)
        if self.use_ssim:  # (1 - ssim_loss) is used to formulate a minimization loss
            loss = (batch_metrics['L1'] + (1 - batch_metrics['SSIM'])) / 2
        else:
            loss = batch_metrics['L1']

        # log metrics
        log_kwargs = {"on_step": False, "on_epoch": True, "batch_size": len(x)}
        self.log(f"loss/{mode}", loss, **log_kwargs)
        for metric_name, value in batch_metrics.items():
            self.log(f"{metric_name}/{mode}", value, **log_kwargs)

        if self.trainer.is_last_batch:
            self.plotting_batch[mode] = x_hat.detach()

        return loss

    def plot_reconstruction(self, mode):
        imgs = self.plotting_batch[mode][:16]
        grid = make_grid(imgs, 4, normalize=True, scale_each=True)
        self.logger.experiment.add_image(f"reconstruction/{mode}", grid, self.current_epoch)

    def training_step(self, batch, batch_idx):
        return self.step_with_monitoring("train", batch, self.train_metrics)

    def on_train_epoch_end(self) -> None:
        self.plot_reconstruction("train")

    def validation_step(self, batch, batch_idx):
        return self.step_with_monitoring("val", batch, self.val_metrics)

    def on_validation_epoch_end(self) -> None:
        self.plot_reconstruction("val")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.parameters())
        return optimizer
