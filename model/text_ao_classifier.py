from clearml import Task
from torch import nn

from model import clip
from model.ao_classifier import AOClassifier


class TxtAOClassifier(AOClassifier):
    def __init__(self):
        super().__init__()
        del self.model
        del self.input_config

        self.txt_encoder = clip.TextEncoder()
        self.latent_dim = self.txt_encoder.latent_dim
        self.tokenizer = self.txt_encoder.tokenizer
        self.classifier = nn.Linear(self.latent_dim, self.classifier[1].out_features)

        self.save_hyperparameters()

    def forward(self, batch):
        # tokenize text
        token_emb = self.tokenizer(batch["report"], return_tensors='pt', padding=True, truncation=True,
                                   max_length=self.tokenizer.model_max_length).data
        input_ids = token_emb['input_ids'].to(self.device, non_blocking=True)
        attention_mask = token_emb['attention_mask'].to(self.device, non_blocking=True)

        features = self.txt_encoder(input_ids=input_ids, attention_mask=attention_mask)

        logits = self.classifier(features)

        return logits

    def on_fit_start(self) -> None:
        self.bce.pos_weight = self.trainer.datamodule.train_dataset.BCE_POS_WEIGHT.to(self.device)
        if Task.current_task() is not None:
            Task.current_task().set_name("TxtAOClassifier")
