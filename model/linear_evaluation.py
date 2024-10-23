from clearml import Task
from torch import nn
from torchvision.models import resnet18

from model.ao_classifier import AOClassifier
from model.clip import LightningCLIP


class ExpanderRGB(nn.Module):
    def forward(self, x):
        return x.expand(-1, 3, -1, -1)


class LinearEvaluation(AOClassifier):
    def __init__(self, type: str = 'CLIP_img'):
        super().__init__()
        del self.model
        del self.input_config
        self.latent_dim = None

        if type == 'CLIP_img':
            ckpt_path = Task.get_task("f91758de8b3b47718225b6f82b728608").artifacts['best.ckpt'].get()
            clip = LightningCLIP.load_from_checkpoint(ckpt_path)
            self.encoder = nn.Sequential(clip.img_encoder, clip.img_projection)
            self.latent_dim = clip.img_projection.projection.out_features
        elif type == 'CLIP_txt':
            ckpt_path = Task.get_task("f91758de8b3b47718225b6f82b728608").artifacts['best.ckpt'].get()
            clip = LightningCLIP.load_from_checkpoint(ckpt_path)
            self.encoder = nn.Sequential(clip.text_encoder, clip.text_projection)
            self.latent_dim = clip.text_projection.projection.out_features
            self.tokenizer = clip.text_encoder.tokenizer
        elif type == 'imagenet':
            resnet = resnet18(weights='DEFAULT')
            self.latent_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.encoder = nn.Sequential(ExpanderRGB(), resnet)
        else:
            raise ValueError(f'Unknown type: {type}')
        self.encoder.requires_grad_(False)
        self.classifier = nn.Linear(self.latent_dim, self.classifier[1].out_features)

        self.save_hyperparameters()

    def forward(self, batch):
        if self.hparams.type in ['imagenet', 'CLIP_img']:  # image processing
            features = self.encoder(batch['image'])
        elif self.hparams.type == 'CLIP_txt':  # text preprocessing
            # tokenize text
            token_emb = self.tokenizer(batch["report"], return_tensors='pt', padding=True, truncation=True,
                                                    max_length=self.tokenizer.model_max_length).data
            input_ids = token_emb['input_ids'].to(self.device, non_blocking=True)
            attention_mask = token_emb['attention_mask'].to(self.device, non_blocking=True)

            features = self.encoder[0](input_ids=input_ids, attention_mask=attention_mask)
            features = self.encoder[1](features)
        else:
            raise ValueError(f'Unknown type: {self.hparams.type}')

        logits = self.classifier(features)

        return logits

    def on_fit_start(self) -> None:
        self.bce.pos_weight = self.trainer.datamodule.train_dataset.BCE_POS_WEIGHT.to(self.device)
        if Task.current_task() is not None:
            Task.current_task().set_name(self.hparams.type)