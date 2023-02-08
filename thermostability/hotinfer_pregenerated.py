from torch import nn
import torch


class HotInferPregeneratedLSTM(nn.Module):
    def __init__(self, hidden_size, hidden_layers):
        super().__init__()
      
        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])
        
        rnn_hidden_size = hidden_size
        rnn_hidden_layers = hidden_layers

        self.thermo_module_rnn = torch.nn.LSTM(input_size=1024,
            hidden_size =rnn_hidden_size, 
            num_layers =rnn_hidden_layers,
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
        output, (hidden, final) = self.thermo_module_rnn(s_s)
        thermostability = self.thermo_module_regression(torch.transpose(hidden, 0,1))
    
        return thermostability
        

class HotInferPregeneratedFC(nn.Module):
    def __init__(self, input_seq_len=700):
        super().__init__()
      
        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])
        
        representation_height = 1024

        self.thermo_module_regression = torch.nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(input_seq_len * representation_height),
            nn.Linear(input_seq_len * representation_height, 1024),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16,1))
            
      

    def forward(self, s_s: torch.Tensor):
        thermostability = self.thermo_module_regression(s_s)
        return thermostability