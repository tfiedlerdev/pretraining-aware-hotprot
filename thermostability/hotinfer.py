from torch import nn
import torch
import esm


class HotInfer(nn.Module):
    def __init__(self, esmfold_config=None, **kwargs):
        super().__init__()
        self.esmfold = esm.pretrained.esmfold_v1()
        self.esmfold = self.esmfold.eval().cuda()
        # cfg = self.esm.cfg
        # s_s shape torch.Size([1, 124, 1024])
        # s_z shape torch.Size([1, 124, 124, 128])
        
        
        #self.thermo_module = nn.Sequential(
        #    nn.LayerNorm(cfg.trunk.structure_module.c_s),
        #    nn.Linear(cfg.trunk.structure_module.c_s, cfg.lddt_head_hid_dim),
        #    nn.ReLU(),
        #    nn.Linear(cfg.lddt_head_hid_dim, cfg.lddt_head_hid_dim),
        #    nn.ReLU(),
        #    nn.Linear(cfg.lddt_head_hid_dim, 37 * self.lddt_bins),
        #    nn.ReLU(),
        #    nn.Linear( 37 * self.lddt_bins,1 ))

    def forward(self,
       sequence:str
        ):
        with torch.no_grad():
            output = self.esmfold.infer(sequences=[sequence])
        print("s_s",str(output["s_s"].size()))
        print("s_z",str(output["s_z"].size()))