import pytorch_lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from torchvision.models import resnet18, resnet34, resnet50


class TextEncoder(nn.Module):
    def __init__(self, huggingface_model_name: str = "distilbert-base-cased", pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(huggingface_model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
        self.tokenizer = DistilBertTokenizer.from_pretrained(huggingface_model_name)
        self.latent_dim = self.model.config.dim

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        sentence_embedding = last_hidden_state[:, self.target_token_idx, :]
        return sentence_embedding


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float = 0):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

# https://github.com/moein-shariatnia/OpenAI-CLIP
class LightningCLIP(L.LightningModule):
    def __init__(self, resnet_depth: int = 18, projection_dim: int = 256, projection_drop_out: float = 0.1,
                 lr: float = 0.001, weight_decay: float = 0.001, temperature_trainable: bool = False,
                 temperature_log_init: float = 1., temperature_clamp: tuple = (1e-5, 100)):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(temperature_log_init).log()).requires_grad_(temperature_trainable)

        # image encoder
        try:
            self.img_encoder = {
                18: resnet18,
                34: resnet34,
                50: resnet50
            }[resnet_depth]
        except KeyError:
            raise NotImplementedError(f"ResNet-{resnet_depth} is not implemented.")
        self.img_encoder = self.img_encoder(weights='DEFAULT')
        # replace first conv to match GrazPedWriDataset
        self.img_encoder.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        img_enc_latent_dim = self.img_encoder.fc.in_features
        self.img_encoder.fc = nn.Identity()
        self.img_projection = ProjectionHead(img_enc_latent_dim, projection_dim, projection_drop_out)

        # text encoder
        self.text_encoder = TextEncoder()
        self.text_encoder.requires_grad_(False)  # freeze text encoder
        self.text_projection = ProjectionHead(self.text_encoder.latent_dim, projection_dim, projection_drop_out)

        self.save_hyperparameters()

    def forward(self, batch):
        # tokenize text
        token_emb = self.text_encoder.tokenizer(batch["report"], return_tensors='pt', padding=True, truncation=True,
                                                max_length=self.text_encoder.tokenizer.model_max_length).data
        input_ids = token_emb['input_ids'].to(self.device, non_blocking=True)
        attention_mask = token_emb['attention_mask'].to(self.device, non_blocking=True)

        img_emb = self.img_encoder(batch["image"])
        img_emb = self.img_projection(img_emb)

        txt_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_emb = self.text_projection(txt_emb)

        return img_emb, txt_emb

    def calculate_loss(self, img_emb, txt_emb):
        temperature = self.logit_scale.exp().clamp(*self.hparams.temperature_clamp)

        logits = (txt_emb @ img_emb.T) / temperature
        img_similarity = img_emb @ img_emb.T
        text_similarity = txt_emb @ txt_emb.T

        y = F.softmax((img_similarity + text_similarity) / (2 * temperature), dim=-1)
        # y = torch.eye(len(img_emb), device=img_emb.device)

        text_loss = F.cross_entropy(logits, y, reduction='none')
        img_loss = F.cross_entropy(logits.T, y, reduction='none')
        loss = (img_loss + text_loss) / 2.0

        return loss.mean(), img_loss.mean(), text_loss.mean()

    def training_step(self, batch, batch_idx):
        img_emb, txt_emb = self(batch)
        loss, img_loss, txt_loss = self.calculate_loss(img_emb, txt_emb)

        log_kwargs = {"on_step": False, "on_epoch": True, "batch_size": len(batch["image"])}
        self.log("CE_image/Train", img_loss.mean().item(), **log_kwargs)
        self.log("CE_text/Train", txt_loss.mean().item(), **log_kwargs)
        self.log("CE_total/Train", loss.mean().item(), **log_kwargs)
        self.log("logit_scale", self.logit_scale.exp().item(), **log_kwargs)

        return loss

    def validation_step(self, batch, batch_idx):
        img_emb, txt_emb = self(batch)
        loss, img_loss, txt_loss = self.calculate_loss(img_emb, txt_emb)

        log_kwargs = {"on_step": False, "on_epoch": True, "batch_size": len(batch["image"])}
        self.log("CE_image/Valid", img_loss.mean().item(), **log_kwargs)
        self.log("CE_text/Valid", txt_loss.mean().item(), **log_kwargs)
        self.log("CE_total/Valid", loss.mean().item(), **log_kwargs)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img_emb, txt_emb = self(batch)

        return {"img_emb": img_emb, "txt_emb": txt_emb, 'img_file': batch['img_file']}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer


if __name__ == '__main__':
    from dataset.grazpedwri_dataset import GrazPedWriDataset
    model = LightningCLIP()
    model({
        "image": torch.randn(2, 1, 224, 224),
        "report": ["This is a test sentence.", "This is another test sentence."]
    })
    ds = GrazPedWriDataset('train')
    reports = [ds.data[file_name]['report'] for file_name in ds.available_file_names]
    token_emb = model.text_encoder.tokenizer(reports, padding=True, truncation=True, max_length=model.text_encoder.tokenizer.model_max_length).data

