from torch import nn
import torch
import esm


class HotInfer(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.esmfold = esm.pretrained.esmfold_v1()
        self.esmfold = self.esmfold.eval().cuda()
        # cfg = self.esm.cfg
        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])
        
        rnn_hidden_size = 128
        rnn_hidden_layers = 2

        thermo_module_rnn = torch.nn.RNN(input_size=1024,
            hidden_size =rnn_hidden_size, 
            num_layers =rnn_hidden_layers, 
            nonlinearity="relu", 
            batch_first =True,
            bidirectional=False)
        
        thermo_module_regression = torch.nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(rnn_hidden_layers * rnn_hidden_size),
            nn.Linear(rnn_hidden_layers * rnn_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16,1))
            
        self.thermo_module_rnn = thermo_module_rnn.to(self.device)
        self.thermo_module_regression = thermo_module_regression.to(self.device)
      

    def forward(self,
       sequence:str
        ):
        with torch.no_grad():
            esm_output = self.esmfold.infer(sequences=[sequence])
            s_s = esm_output["s_s"]

            _, rnn_hidden = self.thermo_module_rnn(s_s)
            
            thermostability = self.thermo_module_regression(torch.transpose(rnn_hidden, 0,1))

        return thermostability
        