from torch import nn
import torch
from esm_custom import esm
from typing import List
import os
import pickle
from thermostability.thermo_pregenerated_dataset import zero_padding_700
from thermostability.hotinfer_pregenerated import HotInferPregeneratedFC
from esm_custom.esm.esmfold.v1.esmfold import RepresentationKey
import csv
class HotInferModel(nn.Module):
    def __init__(
        self,
        representation_key: RepresentationKey,
        thermo_module: nn.Module = HotInferPregeneratedFC(), 
        pad_representations = False,
        model_parallel=False
    ):
        super().__init__()
        self.model_parallel = model_parallel

        self.esmfold = esm.pretrained.esmfold_v1()
        self.thermo_module = thermo_module

        if model_parallel:
            self.esmfold.to("cuda:0")
            self.thermo_module.to("cuda:1")

        self.representation_key = representation_key
        
        self.representations_dir = f"../data/{representation_key}"
        os.makedirs(self.representations_dir, exist_ok=True)
        self.sequences_filepath = os.path.join(self.representations_dir, "sequences.csv")
        if os.path.exists(self.sequences_filepath):
            with open(self.sequences_filepath, "r") as f:
                reader = csv.reader(f, delimiter=",", skipinitialspace=True)
                self.meta = dict([(seq, filename) for i, (seq, filename) in enumerate(reader) if i!=0])
        else:
            self.meta = {}

        self.pad_representations = pad_representations

    def forward(self, sequences: List[str]):
        with torch.no_grad():
            reprs =[]
            for seq in sequences:
                repr = None
                if seq in self.meta:
                    repr = torch.load(
                   os.path.join(self.representations_dir, self.meta[seq])
                )
                else:
                    outputs = self.esmfold.infer(sequences=[seq], representation_key=self.representation_key)
                    repr = outputs[self.representation_key].squeeze()
                    cacheFileName = f"{len(self.meta.keys())+1}.pt"
                    
                    with open(self.sequences_filepath, "a") as f:
                        torch.save(repr, os.path.join(self.representations_dir, cacheFileName))
                        f.write(f"{seq}, {cacheFileName}\n")
                        self.meta[seq] = cacheFileName
                        
                reprs.append(zero_padding_700(repr) if self.pad_representations else repr)
            reprBatch = torch.stack(reprs).to("cuda:1" if self.model_parallel else "cuda:0")
         

        return self.thermo_module(reprBatch)

