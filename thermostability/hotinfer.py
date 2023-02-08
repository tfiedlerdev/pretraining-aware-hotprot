from torch import nn
import torch
import esm
from typing import List
from esmfold_custom import esm as esm_custom
import os
import pickle


class HotInfer(nn.Module):
    def __init__(
        self,
        rnn_hidden_size=128,
        rnn_hidden_layers=2,
        nonlinearity_rnn="relu",
        nonlinearity_thermomodule=nn.ReLU(),
    ):
        super().__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.esmfold = esm.pretrained.esmfold_v1()
        # cfg = self.esm.cfg
        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])

        self.thermo_module_rnn = torch.nn.RNN(
            input_size=1024,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_hidden_layers,
            nonlinearity=nonlinearity_rnn,
            batch_first=True,
            bidirectional=False,
        )

        self.thermo_module_regression = torch.nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(rnn_hidden_layers * rnn_hidden_size),
            nn.Linear(rnn_hidden_layers * rnn_hidden_size, 128),
            nonlinearity_thermomodule,
            nn.Linear(128, 64),
            nonlinearity_thermomodule,
            nn.Linear(64, 16),
            nonlinearity_thermomodule,
            nn.Linear(16, 1),
        )

        # self.thermo_module_rnn = thermo_module_rnn.to(self.device)
        # self.thermo_module_regression = thermo_module_regression.to(self.device)

    def forward(self, sequences: List[str]):
        with torch.no_grad():
            esm_output = self.esmfold.infer(sequences=sequences)
            s_s = esm_output["s_s"]

        _, rnn_hidden = self.thermo_module_rnn(s_s)
        thermostability = self.thermo_module_regression(
            torch.transpose(rnn_hidden, 0, 1)
        )

        return thermostability


class HotInferModelParallel(nn.Module):
    def __init__(
        self,
        rnn_hidden_size=128,
        rnn_hidden_layers=2,
        nonlinearity_rnn="relu",
        nonlinearity_thermomodule=nn.ReLU(),
        representation_key="s_s_0",
    ):
        super().__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.esmfold = esm.pretrained.esmfold_v1().to("cuda:0")
        # cfg = self.esm.cfg
        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])

        self.thermo_module_rnn = torch.nn.RNN(
            input_size=1024,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_hidden_layers,
            nonlinearity=nonlinearity_rnn,
            batch_first=True,
            bidirectional=False,
        ).to("cuda:1")

        self.thermo_module_regression = torch.nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(rnn_hidden_layers * rnn_hidden_size),
            nn.Linear(rnn_hidden_layers * rnn_hidden_size, 128),
            nonlinearity_thermomodule,
            nn.Linear(128, 64),
            nonlinearity_thermomodule,
            nn.Linear(64, 16),
            nonlinearity_thermomodule,
            nn.Linear(16, 1),
        ).to("cuda:1")

        self.representation_key = representation_key
        self.meta_filepath = f"data/model_parallel_{representation_key}_meta.pickle"
        if os.path.exists(self.meta_filepath):
            with open(self.meta_filepath) as f:
                self.meta = pickle.load(f)
        else:
            self.meta = {}
        # self.thermo_module_rnn = thermo_module_rnn.to(self.device)
        # self.thermo_module_regression = thermo_module_regression.to(self.device)

    def forward(self, sequences: List[str]):
        # TODO: optimize (run esmfold for samples while thermomodule runs, replace esmfold by esm2)
        with torch.no_grad():
            reprs = [
                torch.load(
                    os.path.join("data", self.representation_key, self.meta[seq])
                )
                if seq in self.meta
                else self.esmfold.infer(sequences=[seq])[self.representation_key].squeeze()
                for seq in sequences
            ]
            reprBatch = torch.stack(reprs).to("cuda:1")

        _, rnn_hidden = self.thermo_module_rnn(reprBatch)
        thermostability = self.thermo_module_regression(
            torch.transpose(rnn_hidden, 0, 1)
        )

        return thermostability
