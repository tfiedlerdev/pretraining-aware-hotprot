from torch import nn
import torch
from collections import OrderedDict
from typing import Union



def create_fc_layers(
    num: int,
    input_size: int,
    output_size: int,
    p_dropout: float,
    activation=nn.Identity,
) -> nn.Module:
    if num == 1:
        return nn.Sequential(nn.Linear(input_size, output_size), activation())

    result = []
    for i in range(num - 1):
        _output_size = int(input_size / 2)
        layer = nn.Linear(input_size, _output_size)
        result.append((str(i * 3), layer))
        result.append((str(i * 3 + 1), activation()))
        result.append((str(i * 3 + 2), nn.Dropout(p=p_dropout)))
        input_size = _output_size

    result.append((str((num - 1) * 3), nn.Linear(input_size, output_size)))
    result.append((str((num - 1) * 3 + 1), activation()))
    return nn.Sequential(OrderedDict(result))


class HotInferPregeneratedLSTM(nn.Module):
    def __init__(self, hidden_size, hidden_layers):
        super().__init__()

        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])

        rnn_hidden_size = hidden_size
        rnn_hidden_layers = hidden_layers

        self.thermo_module_rnn = torch.nn.LSTM(
            input_size=1024,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_hidden_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.thermo_module_regression = torch.nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(rnn_hidden_layers * rnn_hidden_size),
            nn.Linear(rnn_hidden_layers * rnn_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, s_s: torch.Tensor):
        output, (hidden, final) = self.thermo_module_rnn(s_s)
        thermostability = self.thermo_module_regression(torch.transpose(hidden, 0, 1))

        return thermostability


class HotInferPregeneratedFC(nn.Module):
    def __init__(
        self, input_len=700, num_hidden_layers=3, first_hidden_size=1024, p_dropout=0.0
    ):
        super().__init__()

        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])

        self.thermo_module_regression = torch.nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(input_len),
            nn.Linear(input_len, first_hidden_size),
            nn.ReLU(),
            create_fc_layers(
                num_hidden_layers, first_hidden_size, 16, p_dropout, activation=nn.ReLU
            ),
            nn.Linear(16, 1),
        )

    def forward(self, s_s: torch.Tensor):
        thermostability = self.thermo_module_regression(s_s)
        return thermostability


class HotInferPregeneratedSummarizerFC(nn.Module):
    def __init__(
        self,
        summarizer: nn.Module,
        thermo_module: nn.Module,
        p_dropout=0.0,
    ):
        super().__init__()

        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])

        self.summarizer = summarizer
        self.thermo_module_regression = thermo_module

    def forward(self, s_s: torch.Tensor):
        thermostability = self.thermo_module_regression(self.summarizer(s_s))
        return thermostability
