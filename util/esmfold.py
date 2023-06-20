from typing import Any
import torch
from esm_custom import esm
from esm_custom.esm.esmfold.v1.esmfold import RepresentationKey


class ESMFoldEmbeddings:
    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.device = device

        if not torch.cuda.is_available():
            print(
                "WARNING: A Cuda supporting GPU is not available. This will likely fail or take ages"
            )
        elif (
            torch.cuda.get_device_properties(self.device).total_memory
            < 1000 * 1000 * 1000 * 35
        ):
            print(
                "WARNING: Cuda device at position 0 has less than 35GB of memory. This was only tested with Nvidia A40 with 40GB+ of memory"
            )

        self.esmfold = esm.pretrained.esmfold_v1().to(self.device)

    def __call__(self, sequences, representation_key: RepresentationKey = "s_s") -> Any:
        return self.esmfold.infer(
            sequences=sequences, representation_key=representation_key
        )[representation_key]
