from torch import nn
import torch
from typing import List
import os
import pickle


class DenseModel(nn.Module):
    def __init__(
        self,
        rnn_hidden_size=128,
        nonlinearity_rnn="relu",
        nonlinearity_thermomodule=nn.ReLU(),
    ):
        super().__init__()
        # input shape [1024]
        input_shape = 1024
        self.thermo_module_regression = torch.nn.Sequential(
            nn.LayerNorm(input_shape),
            nn.Linear(input_shape, rnn_hidden_size),
            nonlinearity_thermomodule,
            nn.Linear(rnn_hidden_size, 64),
            nonlinearity_thermomodule,
            nn.Linear(64, 16),
            nonlinearity_thermomodule,
            nn.Linear(16, 1),
        )
        #self.thermo_module_regression = self.thermo_module_regression.to('cuda:0')

    def forward(self, representations: torch.Tensor):
        converted_representation = representations.to(torch.float32)
        thermostability = self.thermo_module_regression(converted_representation)
        return thermostability