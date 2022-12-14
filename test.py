from thermostability.hotinfer import HotInfer
import torch
import typing as T
from esm_module.esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)


hot = HotInfer()

fasta = "MSALFQEVKGRQQDFMKAFNAGDAAGAASVYDPDGYFMPNGRNPVKGRSGIEAYFKEDMADGVQTAQIITEEVNGGGDWAFERGSYHLDGTKGRESGAYLQIWKKVEGVWLIHNDCFNVIKNAC"
aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(sequences= [fasta])
residx = None
if residx is None:
    residx = _residx
elif not isinstance(residx, torch.Tensor):
    residx = collate_dense_tensors(residx)

aatype, mask, residx, linker_mask = map(
    lambda x: x.to(aatype.device), (aatype, mask, residx, linker_mask)
)
        
hot.forward(aatype,
            mask=mask,
            residx=residx,
          )