from torch import nn
import torch
import esm
from typing import List

class HotInferPregenerated(nn.Module):
    def __init__(
        self,
        rnn_hidden_size = 128,
        rnn_hidden_layers = 1,
        ):
        super().__init__()

        self.thermo_module_rnn = torch.nn.RNN(input_size=1024,
            hidden_size =rnn_hidden_size, 
            num_layers =rnn_hidden_layers, 
            nonlinearity="relu", 
            batch_first =True,
            bidirectional=False)
        
        self.thermo_module_regression = torch.nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(rnn_hidden_layers * rnn_hidden_size),
            nn.Linear(rnn_hidden_layers * rnn_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16,1))
            
      

    def forward(self, s_s: torch.Tensor):
        _, rnn_hidden = self.thermo_module_rnn(s_s)
        thermostability = self.thermo_module_regression(torch.transpose(rnn_hidden, 0,1))
    
        return thermostability
        