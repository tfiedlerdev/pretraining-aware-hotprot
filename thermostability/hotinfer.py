from torch import nn
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import torch
from esm_module.esm.esmfold.v1.misc import batch_encode_sequences
from openfold.openfold.np import residue_constants
import typing as T
from esm_module.esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)



model_location = "esm2_t36_3B_UR50D"
fasta_file = ""
toks_per_batch = 4096
include = ["per_tok"]
truncation_seq_length = 1022
no_gpu = False

class HotInfer(nn.Module):
    def __init__(self, esmfold_config=None, **kwargs):
        self.esm, self.esm_dict = pretrained.esm2_t33_650M_UR50D()
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
        aa: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = None,):
        
        # Passing sequence through ESM and creating representations

        if mask is None:
            mask = torch.ones_like(aa)

        B = aa.shape[0]
        L = aa.shape[1]
        device = aa.device

        if residx is None:
            residx = torch.arange(L, device=device).expand_as(aa)

        # === ESM ===
        esmaa = self._af2_idx_to_esm_idx(aa, mask)

        if masking_pattern is not None:
            esmaa = self._mask_inputs_to_esm(esmaa, masking_pattern)

        esm_s = self._compute_language_model_representations(esmaa)

        # esmOut = self.esm(aa, mask, residx, masking_pattern, num_recycles)

        return esm_s.shape
        # return F.relu(self.conv2(x))




    def _compute_language_model_representations(self, esmaa: torch.Tensor) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=False,
        )
        esm_s = torch.stack([v for _, v in sorted(res["representations"].items())], dim=2)
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        return esm_s

    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [d.get_idx(v) for v in residue_constants.restypes_with_x]
        return torch.tensor(esm_reorder)

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]