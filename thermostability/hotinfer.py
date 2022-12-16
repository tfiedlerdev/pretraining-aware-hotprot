from torch import nn
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import torch
from esm.esmfold.v1.misc import batch_encode_sequences
from openfold.np import residue_constants
import typing as T
from esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)
import esm



model_location = "esm2_t36_3B_UR50D"
fasta_file = ""
toks_per_batch = 4096
include = ["per_tok"]
truncation_seq_length = 1022
no_gpu = False

class HotInfer(nn.Module):
    def __init__(self, esmfold_config=None, **kwargs):
        super.__init__(self)
        self.esmfold = esm.pretrained.esmfold_v1()
        self.esmfold = self.esmfold.eval().cuda()
        # cfg = self.esm.cfg

        # self.thermo_module = nn.Sequential(
        #     nn.LayerNorm(cfg.trunk.structure_module.c_s),
        #     nn.Linear(cfg.trunk.structure_module.c_s, cfg.lddt_head_hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(cfg.lddt_head_hid_dim, cfg.lddt_head_hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(cfg.lddt_head_hid_dim, 37 * self.lddt_bins),
        #     nn.ReLU(),
        #     nn.Linear( 37 * self.lddt_bins,1 ))

    def forward(self,
       sequence:str
        ):
        with torch.no_grad():
            output = self.esmfold.infer_pdb(sequence)
        print(str(output))