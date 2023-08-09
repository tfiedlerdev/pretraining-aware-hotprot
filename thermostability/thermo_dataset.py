from torch.utils.data import Dataset
import torch
import os
import csv
import numpy as np
from typing import Literal


def calc_norm(temps: "list[float]"):
    temps_np = np.array(temps)
    return temps_np.mean(), temps_np.var()


datasets = {
    "ours": {
        "train": "data/train.csv",
        "test": "data/test.csv",
        "val": "data/val.csv",
    },
    "ours_median": {
        "train": "data/train_median.csv",
        "test": "data/test_median.csv",
        "val": "data/val_median.csv",
    },
    "flip": {
        "train": "data/train_flip.csv",
        "test": "data/test_flip.csv",
        "val": "data/val_flip.csv",
    },
}
DatasetNames = Literal["ours", "ours_median", "flip"]


class ThermostabilityDataset(Dataset):
    def __init__(
        self,
        dataset: DatasetNames,
        split: Literal["train", "test", "val"],
        limit: int = 100000,
        max_seq_len: int = 700,
    ) -> None:
        super().__init__()
        dataset_filepath = datasets[dataset][split]
        if not os.path.exists(dataset_filepath):
            raise Exception(f"{dataset_filepath} does not exist.")

        self.limit = limit
        with open(dataset_filepath, newline="\n") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", skipinitialspace=True)
            self.seq_thermos = [
                (seq, thermo)
                for (i, (seq, thermo)) in enumerate(spamreader)
                if i != 0 and len(seq) <= max_seq_len
            ]

    def norm_distr(self):
        temps = [float(thermo) for (seq, thermo) in self.seq_thermos]
        return calc_norm(temps)

    def __len__(self):
        return min(len(self.seq_thermos), self.limit)

    def __getitem__(self, index):
        seq, thermo = self.seq_thermos[index]
        return seq, torch.tensor(float(thermo), dtype=torch.float32)
