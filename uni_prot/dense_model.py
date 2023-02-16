from collections import OrderedDict
from torch import nn
import torch
from typing import List
import os
import pickle


class DenseModel(nn.Module):
    def __init__(
        self,
        rnn_hidden_size=128,
        layers = 4,
        dropout_rate = 0.3,
        nonlinearity_rnn="relu",
        nonlinearity_thermomodule=nn.ReLU(),
    ):
        super().__init__()
        # input shape [1024]
        self.input_shape = 1024
        self.dropout_rate = dropout_rate

        hidden_layers = self.create_hidden_layers(layers, self.input_shape*3, 16)

        self.thermo_module_regression = torch.nn.Sequential(
            nn.LayerNorm(self.input_shape),
            hidden_layers,
            nn.Linear(16, 1),
        )
        #self.thermo_module_regression = self.thermo_module_regression.to('cuda:0')

    def create_hidden_layers(self,num:int, input_size:int, output_size:int) -> nn.Module:
        if num == 1:
            return nn.Sequential(nn.Linear(self.input_shape, output_size), nn.ReLU())
        
        result = []
        result.append((str(0),nn.Linear(self.input_shape, input_size)))
        result.append((str(1),nn.ReLU()))
        result.append((str(2),nn.Dropout(p=self.dropout_rate)))

        for i in range(1, num-1):
            print(input_size)
            _output_size = int(input_size/2)
            layer = nn.Linear(input_size, _output_size)
            result.append((str(i*3),layer))
            result.append((str(i*3+1),nn.ReLU()))
            result.append((str(i*3+2),nn.Dropout(p=self.dropout_rate)))
            input_size = _output_size
        
        result.append((str((num-1)*3),nn.Linear(input_size, output_size)))
        result.append((str((num-1)*3+1),nn.ReLU()))

        return nn.Sequential(OrderedDict(result))


    def forward(self, representations: torch.Tensor):
        converted_representation = representations.to(torch.float32)
        thermostability = self.thermo_module_regression(converted_representation)
        return thermostability