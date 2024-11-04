from clearml import Task
from torch import nn
from torchvision.models import resnet18

from model.ao_classifier import AOClassifier
from model.clip import LightningCLIP
from model.autoencoder import LightningAutoEncoder


# helper to fake a 3 channel image given a single channel image
class ExpanderRGB(nn.Module):
    def forward(self, x):
        return x.expand(-1, 3, -1, -1)


class AEAvgPool(nn.Module):
    def __init__(self, img_encoder: nn.Module, latent_dim: int):
        """
        Spatial average pooling. Collapse spatial dimensions to 1
        :param img_encoder: Encoder to collapse output of
        :param latent_dim: number of channels of the encoder output
        """
        super().__init__()
        self.img_encoder = img_encoder
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.img_encoder(x)
        x = x.view(x.shape[0], self.latent_dim, -1).mean(-1)
        return x


class LinearEvaluation(AOClassifier):
    def __init__(self, type: str = 'CLIP_img'):
        """
        Linear evaluation model: Fits a single linear layer to the frozen features of a pretrained model.
        :param type: Type of model to use as encoder. Options: 'CLIP_img', 'CLIP_txt', 'imagenet', 'AE'
        """
        super().__init__()
        del self.model
        del self.input_config
        self.latent_dim = None

        if type == 'CLIP_img':
            task_id = 'f91758de8b3b47718225b6f82b728608'
            print(f'Loading model from task {task_id}')
            ckpt_path = Task.get_task(task_id).artifacts['best.ckpt'].get()
            clip = LightningCLIP.load_from_checkpoint(ckpt_path)
            self.encoder = nn.Sequential(clip.img_encoder, clip.img_projection)
            self.latent_dim = clip.img_projection.projection.out_features
        elif type == 'CLIP_txt':
            task_id = 'f91758de8b3b47718225b6f82b728608'
            print(f'Loading model from task {task_id}')
            ckpt_path = Task.get_task(task_id).artifacts['best.ckpt'].get()
            clip = LightningCLIP.load_from_checkpoint(ckpt_path)
            self.encoder = nn.Sequential(clip.text_encoder, clip.text_projection)
            self.latent_dim = clip.text_projection.projection.out_features
            self.tokenizer = clip.text_encoder.tokenizer
        elif type == 'imagenet':
            resnet = resnet18(weights='DEFAULT')
            self.latent_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.encoder = nn.Sequential(ExpanderRGB(), resnet)
        elif type == 'AE':
            task_id = '9d3f7de80bda411d9662cfd6ef793f4d'
            print(f'Loading model from task {task_id}')
            ckpt_path = Task.get_task(task_id).artifacts['best.ckpt'].get()
            ae = LightningAutoEncoder.load_from_checkpoint(ckpt_path)
            self.latent_dim = ae.latent_dim
            self.encoder = AEAvgPool(ae.img_encoder, self.latent_dim)
        else:
            raise ValueError(f'Unknown type: {type}')
        self.encoder.requires_grad_(False)
        self.classifier = nn.Linear(self.latent_dim, self.classifier[1].out_features)

        self.save_hyperparameters()

    def forward(self, batch):
        if self.hparams.type in ['imagenet', 'CLIP_img', 'AE']:  # image processing
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
