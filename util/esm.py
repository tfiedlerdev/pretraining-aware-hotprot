from typing import Any
import torch
from esm_custom.esm.pretrained import load_model_and_alphabet_hub
from esm_custom.esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
)
from esm_custom.esm.esmfold.v1.esmfold import RepresentationKey
from openfold.np import residue_constants

def preprocess_sequences(sequences: "list[str]", alphabet, device: str = "cuda:0"):
    aatype, mask, residx, linker_mask, chain_index = batch_encode_sequences(sequences)
    
    if not isinstance(residx, torch.Tensor):
        residx = collate_dense_tensors(residx)
        
    aatype, mask, residx, linker_mask = map(
        lambda x: x.to(device), (aatype, mask, residx, linker_mask)
    )

    B = aatype.shape[0]
    L = aatype.shape[1]
    device = aatype.device

    # === ESM ===
    aatype = (aatype + 1).masked_fill(mask != 1, 0)
    af2_to_esm = torch.tensor([alphabet.padding_idx] + [alphabet.get_idx(v) for v in residue_constants.restypes_with_x]).to(device)
    esmaa = af2_to_esm[aatype]

    batch_size = esmaa.size(0)
    bosi, eosi = alphabet.cls_idx, alphabet.eos_idx
    bos = esmaa.new_full((batch_size, 1), bosi)
    eos = esmaa.new_full((batch_size, 1), alphabet.padding_idx)
    esmaa = torch.cat([bos, esmaa, eos], dim=1)
    # Use the first padding index as eos during inference.
    esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi
    
    return esmaa


class ESMEmbeddings:
    def __init__(self, model_id: str, device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        self.device = device
        self.esm, self.alphabet = load_model_and_alphabet_hub(model_id)
        self.esm.to(self.device)
        
    def __call__(self, sequences: "list[str]", representation_key: RepresentationKey) -> Any:
        esmaa = preprocess_sequences(sequences, self.alphabet)
        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=False,
        )
        
        esm_s = torch.stack([v for _, v in sorted(res["representations"].items())], dim=2)
        esm_s = esm_s[:, 1:-1]
        fst_output = torch.mean(esm_s, dim=1)
        return fst_output