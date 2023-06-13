import torch
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmModel
from torch import nn
from thermostability.hotinfer import CachedModel, RepresentationKeysComb
from thermostability.hotinfer_pregenerated import create_fc_layers


class ESMForThermostability(CachedModel):
    def __init__(
        self,
        regressor_layers: int = 3,
        regressor_dropout: float = 0.3,
        regressor_activation: nn.Module = nn.LeakyReLU(),
    ):
        super().__init__("start_token", True)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.regression = create_fc_layers(
            num=regressor_layers,
            input_size=1024,
            p_dropout=regressor_dropout,
            activation=regressor_activation,
        ).to("cuda:0")

    def _get_esm(self):
        if not self.esm:
            self.esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(
                "cuda:0"
            )
        return self.esm

    def forward(self, sequences: "list[str]"):
        esm = self._get_esm()
        input_ids = self.tokenizer(
            sequences, padding=True, truncation=True, return_tensors="pt"
        ).input_ids.to("cuda:0")
        outputs = esm(input_ids.to("cuda:0"))
        last_hidden_state = outputs.last_hidden_state
        return self.regression(last_hidden_state[:, 0, :])

    def compute_representation(self, seq: str, _: RepresentationKeysComb):
        esm = self._get_esm()
        input_ids = self.tokenizer(
            [seq], padding=True, truncation=True, return_tensors="pt"
        ).to("cuda:0")
        outputs = esm(input_ids.to("cuda:0"))
        last_hidden_state = outputs.last_hidden_state

        s_embedding = last_hidden_state[:, 0, :]
        return s_embedding
