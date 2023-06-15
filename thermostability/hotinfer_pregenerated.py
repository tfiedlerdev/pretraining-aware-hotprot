from torch import nn
import torch
from collections import OrderedDict


def create_fc_layers(
    num: int,
    input_size: int,
    output_size: int,
    p_dropout: float,
    activation=nn.Identity,
    use_layer_norm_before_first=False,
) -> nn.Module:
    if num == 1:
        if not use_layer_norm_before_first:
            return nn.Sequential(nn.Linear(input_size, output_size), activation())
        else:
            return nn.Sequential(
                nn.LayerNorm(input_size),
                nn.Linear(input_size, output_size),
                activation(),
            )

    result = []
    id = lambda: str(len(result))
    for i in range(num - 1):
        _output_size = int(input_size / 2)
        if i == 0 and use_layer_norm_before_first:
            result.append((id(), nn.LayerNorm(input_size)))
        layer = nn.Linear(input_size, _output_size)
        result.append((id(), layer))
        result.append((id(), activation()))
        result.append((id(), nn.Dropout(p=p_dropout)))
        input_size = _output_size

    result.append((id(), nn.Linear(input_size, output_size)))
    result.append((id(), activation()))
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
