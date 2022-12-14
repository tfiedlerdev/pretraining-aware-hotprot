from esm.esmfold import ESMFold 
from torch import nn

class HotInfer(nn.Module):
    def __init__(self, esmfold_config=None, **kwargs):
        self.esm = ESMFold()
        cfg = self.esm.cfg
        self.esm.lddt_head = nn.Sequential(
            nn.LayerNorm(cfg.trunk.structure_module.c_s),
            nn.Linear(cfg.trunk.structure_module.c_s, cfg.lddt_head_hid_dim),
            
            nn.Linear(cfg.lddt_head_hid_dim, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, 37 * self.lddt_bins),)

