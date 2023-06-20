import torch
from tqdm import tqdm
from torch.nn.modules import Module
import torch.nn.utils.prune as prune
from typing import Callable
from esm_custom.esm.esmfold.v1.pretrained import esmfold_v1
from esm_custom.esm.esmfold.v1.esmfold import ESMFold
from esm_custom.esm.sparse_multihead_attention import SparseMultiheadAttention

class FSTHotProt(Module):
    def __init__(self, hotprot_model, padding: Callable, factorized_sparse_tuning_rank: int = 4, sparse: int = 64):
        super().__init__()
        self.hotinfer = hotprot_model
        self.padding = padding
        esm_fold = esmfold_v1()
        self.fst_esm = ESMFold(esmfold_config=esm_fold.cfg, use_sparse=True, rank=factorized_sparse_tuning_rank).to("cuda:0")
        self.fst_esm.load_state_dict(esm_fold.state_dict(), strict=False)
        del esm_fold
        
        for name, m in self.fst_esm.named_modules():
            if "adapter" in name or "sparse" in name:
                m.requires_grad = True
            else:
                m.requires_grad = False
                
        for name, m in self.fst_esm.named_modules():
            if isinstance(m, SparseMultiheadAttention):
                Q_weight = m.q_proj.weight
                V_weight = m.v_proj.weight
                Q_weight = Q_weight.detach().cpu().float()
                V_weight = V_weight.detach().cpu().float()
                U_Q = torch.randn((Q_weight.shape[0], 1)).to(Q_weight.device)
                V_Q = torch.randn((1, Q_weight.shape[1])).to(Q_weight.device)
                S_Q = torch.zeros_like(Q_weight)

                U_V = torch.randn((V_weight.shape[0], 1)).to(V_weight.device)
                V_V = torch.randn((1, V_weight.shape[1])).to(V_weight.device)
                S_V = torch.zeros_like(V_weight)
                last_S_Q = torch.zeros_like(Q_weight)

                for rank in tqdm(range(20)):
                    S_Q = torch.zeros_like(Q_weight)
                    S_V = torch.zeros_like(Q_weight)
                    for _ in range(10):
                        #print(_, residual_change)
                        U_Q = torch.qr((Q_weight - S_Q) @ V_Q.T)[0]
                        V_Q = U_Q.T @ (Q_weight - S_Q)
                        S_Q = Q_weight - U_Q @ V_Q
                        q = 0.01
                        S_Q[S_Q.abs() < q] = 0
                        # residual_change = torch.norm(S_Q - last_S_Q, p=2)
                        last_S_Q = S_Q
                        
                        U_V = torch.qr((V_weight - S_V) @ V_V.T)[0]
                        V_V = U_V.T @ (V_weight - S_V)
                        S_V = V_weight - U_V @ V_V
                        #residual_change.append(torch.norm(Q_weight - U_V@V_V).item())
                        q = 0.01
                        S_V[S_V.abs() < q] = 0

                    E_Q = Q_weight - U_Q @ V_Q - S_Q
                    E_V = V_weight - U_V @ V_V - S_V
                    
                    E_Q_vector = torch.qr(E_Q)[1][:1]
                    E_V_vector = torch.qr(E_V)[1][:1]
                    
                    V_Q = torch.cat([V_Q, E_Q_vector], 0)
                    V_V = torch.cat([V_V, E_V_vector], 0)
                
                q, _ = torch.kthvalue(S_Q.abs().view(-1), S_Q.numel() - sparse)
                S_Q = (S_Q.abs() >= q).float()
                #print(S_Q)
                v, _ = torch.kthvalue(S_V.abs().view(-1), S_V.numel() - sparse)
                S_V = (S_V.abs() >= v).float()
                prune.custom_from_mask(m.q_proj_sparse, 'weight', S_Q.to(m.q_proj.weight.device))
                prune.custom_from_mask(m.v_proj_sparse, 'weight', S_V.to(m.v_proj.weight.device))
                
    def forward(self, input: "tuple[list[str], torch.Tensor]"):
        sequences, esm_embeddings = input
        
        fst_embeddings = self.fst_esm.infer(sequences=sequences)["s_s"]

        fst_input = torch.stack([self.padding(emb) for emb in fst_embeddings])

        return self.hotinfer(torch.add(fst_input, esm_embeddings))
        