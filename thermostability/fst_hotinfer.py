import torch
from tqdm import tqdm
import torch.nn.utils.prune as prune
from esm_custom.esm.pretrained import load_model_and_alphabet_hub
from esm_custom.esm.sparse_multihead_attention import SparseMultiheadAttention
from util.esm import preprocess_sequences
from os.path import exists
from pathlib import Path
from torch.nn import Module
from thermostability.thermo_pregenerated_dataset import zero_padding_700
from util.esm import ESMModelType

class FSTHotProt(Module):
    def __init__(self, hotprot_model, esm_model: ESMModelType = "esm2_t33_650M_UR50D", factorized_sparse_tuning_rank: int = 4, sparse: int = 64):
        super().__init__()
        self.hotinfer = hotprot_model

        model_cache_path = f"cache/fst_{esm_model}_{factorized_sparse_tuning_rank}.pt"
        alphabet_cache_path = f"cache/alphabet_{esm_model}_{factorized_sparse_tuning_rank}.pt"
        if exists(model_cache_path) and exists(alphabet_cache_path):
            self.fst_esm = torch.load(model_cache_path)
            self.alphabet = torch.load(alphabet_cache_path)
            print("Loaded cached and untrained FST Hotinfer model")
        else:
            self.fst_esm, self.alphabet = load_model_and_alphabet_hub(esm_model, True, factorized_sparse_tuning_rank)
            
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
                    U_Q = torch.randn((Q_weight.shape[0], 1))
                    V_Q = torch.randn((1, Q_weight.shape[1]))
                    S_Q = torch.zeros_like(Q_weight)

                    U_V = torch.randn((V_weight.shape[0], 1))
                    V_V = torch.randn((1, V_weight.shape[1]))
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
                    prune.custom_from_mask(m.q_proj_sparse, 'weight', S_Q)
                    prune.custom_from_mask(m.v_proj_sparse, 'weight', S_V)

            if not Path(model_cache_path).parent.is_dir():
                Path(model_cache_path).parent.mkdir(parents=True)
            with open(model_cache_path, "wb") as f:
                torch.save(self.fst_esm.cpu(), f)
            with open(alphabet_cache_path, "wb") as f:
                torch.save(self.alphabet, f)
     
    def calculate_representations(self, sequences: "list[str]"):
        esmaa = preprocess_sequences(sequences, self.alphabet)
        res = self.fst_esm(
            esmaa,
            repr_layers=range(self.fst_esm.num_layers + 1),
            need_head_weights=False,
        )
        esm_s = torch.stack([v for _, v in sorted(res["representations"].items())], dim=2)
        esm_s = esm_s[:, 1:-1]
        fst_output = torch.mean(esm_s, dim=2)
        return fst_output
    
    def dummy_test(self, sequences: "list[str]"):
        """
        Can be used to test if the model runs if embeddings have not been pregenerated
        """
        fst_output = self.calculate_representations(sequences)
        return self.hotinfer(zero_padding_700(fst_output, dim=1))
                
    def forward(self, input: "tuple[list[str], torch.Tensor]"):
        sequences, esm_embeddings = input
        fst_output = self.calculate_representations(sequences)
        fst_output = zero_padding_700(fst_output, dim=1)
        return self.hotinfer(torch.add(fst_output, esm_embeddings))
        