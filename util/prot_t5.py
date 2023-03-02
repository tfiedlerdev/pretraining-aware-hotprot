from typing import Any
import torch
from transformers import T5Tokenizer, T5EncoderModel
import regex as re


class ProtT5Embeddings:
    # Source: https://github.com/agemagician/ProtTrans#quick
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
        )

        # Load the model
        self.model = T5EncoderModel.from_pretrained(
            "Rostlab/prot_t5_xl_half_uniref50-enc"
        ).to(self.device)

        # only GPUs support half-precision currently; if you want to run on CPU
        # use full-precision (not recommended, much slower)
        self.model.full() if self.device == "cpu" else self.model.half()

    def __call__(self, sequences) -> Any:
        # replace all rare/ambiguous amino acids by X and introduce white-
        # space between all amino acids
        sequences = [
            " ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences
        ]

        # tokenize sequences and pad up to the longest sequence in the batch
        ids = self.tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding="longest",
        )

        input_ids = torch.tensor(ids["input_ids"]).to(self.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self.device)

        # generate embeddings
        with torch.no_grad():
            embedding_rpr = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )

        # extract residue embeddings for the first ([0,:]) sequence in the
        # batch and remove padded & special tokens ([0,:7])
        emb_0 = embedding_rpr.last_hidden_state[0, :7]  # shape (7 x 1024)
        # same for the second ([1,:]) sequence but taking into account
        # different sequence lengths ([1,:8])
        # emb_1 = embedding_rpr.last_hidden_state[1, :8]  # shape (8 x 1024)

        
        return emb_0
