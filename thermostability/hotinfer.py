from torch import nn
import torch
from typing import List, Literal, Union, Optional
import os
from thermostability.thermo_pregenerated_dataset import zero_padding_700
from thermostability.hotinfer_pregenerated import HotInferPregeneratedFC
from esm_custom.esm.esmfold.v1.esmfold import RepresentationKey
import csv
from util.prot_t5 import ProtT5Embeddings
from util.esmfold import ESMFoldEmbeddings
from util.esm import ESMEmbeddings
from abc import ABC, abstractmethod

RepresentationKeysComb = Union[
    RepresentationKey, Literal["prott5_avg", "prott5", "esm_3B_avg"]
]


class CachedModel(nn.Module, ABC):
    def __init__(
        self,
        representation_key: RepresentationKeysComb,
        caching=True,
        enable_grad=False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        assert (
            type(caching) == bool
        ), f"Caching must be a boolean but is {type(caching)}"
        if caching:
            assert (
                enable_grad == False
            ), "Enable grad can only be true if caching is disabled"

        print(
            "Initializing CachedModel with caching: ",
            caching,
            " and enable_grad: ",
            enable_grad,
        )
        self._enable_grad = enable_grad
        self.representation_key = representation_key
        self._caching = caching
        self.representations_dir = (
            f"./data/{representation_key}"
            if cache_dir is None
            else os.path.join(cache_dir, representation_key)
        )

        if caching:
            os.makedirs(self.representations_dir, exist_ok=True)
        self.sequences_filepath = os.path.join(
            self.representations_dir, "sequences.csv"
        )
        if os.path.exists(self.sequences_filepath) and caching:
            with open(self.sequences_filepath, "r") as f:
                reader = csv.reader(f, delimiter=",", skipinitialspace=True)
                self.meta = dict(
                    [
                        (seq, filename)
                        for i, (seq, filename) in enumerate(reader)
                        if i != 0
                    ]
                )
        else:
            self.meta = {}

    def get_cached_or_compute(self, sequences: "list[str]"):
        with torch.set_grad_enabled(self._enable_grad):
            if self._caching:
                reprs = []
                for seq in sequences:
                    repr = None
                    if seq in self.meta:
                        repr = torch.load(
                            os.path.join(self.representations_dir, self.meta[seq])
                        )
                        if type(repr) == list:
                            repr = torch.stack(repr)
                    else:
                        repr = self.compute_representations(
                            [seq], self.representation_key
                        )
                        cacheFileName = f"{len(self.meta.keys())+1}.pt"
                        with open(self.sequences_filepath, "a") as f:
                            torch.save(
                                repr,
                                os.path.join(self.representations_dir, cacheFileName),
                            )
                            f.write(f"{seq}, {cacheFileName}\n")
                            self.meta[seq] = cacheFileName
                    reprs.append(self.prepare_repr_before_collate(repr))
                reprBatch = torch.stack(reprs)
                return reprBatch

            return self.compute_representations(sequences, self.representation_key)

    def prepare_repr_before_collate(self, repr: torch.Tensor):
        return repr

    @abstractmethod
    def compute_representations(
        self, seq: str, representation_key: RepresentationKeysComb
    ) -> torch.Tensor:
        pass


class HotInferModel(CachedModel):
    def __init__(
        self,
        representation_key: RepresentationKeysComb,
        thermo_module: nn.Module = HotInferPregeneratedFC(),
        pad_representations=False,
        model_parallel=False,
        caching=True,
    ):
        super().__init__(representation_key, caching)
        self.model_parallel = model_parallel
        self.caching = caching

        self.repr_model = (
            ProtT5Embeddings(device="cuda:0")
            if representation_key == "prott5_avg"
            else ESMEmbeddings(device="cuda:0")
        )

        self.thermo_module = thermo_module

        if model_parallel:
            self.thermo_module.to("cuda:1")

        self.pad_representations = pad_representations

    def forward(self, sequences: List[str]):
        reprBatch = self.get_cached_or_compute(sequences).to(
            "cuda:1" if self.model_parallel else "cuda:0"
        )

        return self.thermo_module(reprBatch)

    def compute_representations(
        self, seqs: "list[str]", representation_key: RepresentationKeysComb
    ):
        return self.repr_model(sequences=seqs, representation_key=representation_key)

    def prepare_repr_before_collate(self, repr: torch.Tensor):
        return zero_padding_700(repr) if self.pad_representations else repr
