from torch.utils.data import Dataset
import torch
import os
import csv
from typing import Union
from torch.nn.functional import pad
from thermostability.thermo_dataset import calc_norm
from esm_custom.esm.esmfold.v1.esmfold import RepresentationKey
from thermostability.thermo_pregenerated_dataset import (
    ThermostabilityPregeneratedDataset,
)
from thermostability.thermo_pregenerated_dataset import zero_padding700_collate


def zero_padding_fst(seq_list: "list[tuple[str, torch.Tensor, torch.Tensor]]"):
    seqs = [seq for seq, emb, temp in seq_list]
    return seqs, zero_padding700_collate([(emb, temp) for seq, emb, temp in seq_list])


class FSTDataset(ThermostabilityPregeneratedDataset):
    def __init__(
        self,
        dsFilePath: str = "data/train.csv",
        limit: int = 1000000,
        max_seq_len: int = 700,
        representation_filepath: str = "data",
        representation_key: RepresentationKey = "s_s_avg",
    ) -> None:
        super().__init__(
            dsFilePath=dsFilePath,
            limit=limit,
            max_seq_len=max_seq_len,
            representation_filepath=representation_filepath,
            representation_key=representation_key,
        )

    def __getitem__(self, index):
        filename, thermo, seq = self.filename_thermo_seq[index]
        with open(os.path.join(self.representations_dir, filename), "rb") as f:
            s_s = torch.load(f)

        return seq, s_s, torch.tensor(thermo, dtype=torch.float32)
