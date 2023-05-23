from torch import nn
import torch
from collections import OrderedDict


class CNNPregenerated(nn.Module):
    def __init__(self, hidden_size, hidden_layers):
        super.__init__()
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

        self.thermo_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
        )

    def forward(self, s_s: torch.Tensor):
        _, (hidden, _) = self.thermo_module_rnn(s_s)
        thermostability = self.thermo_module_regression(torch.transpose(hidden, 0, 1))
        return thermostability


class CNNPregeneratedFC(nn.Module):
    def __init__(self, input_seq_len=700, num_hidden_layers=1, first_hidden_size=512):
        super().__init__()

        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])
        output_channels = 16

        self.thermo_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(
                in_channels=8,
                out_channels=output_channels,
                kernel_size=3,
                stride=(1, 1),
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=(1, 1),
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=(1, 1),
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        inter_height = 41
        inter_width = 62

        self.thermo_module_regression = torch.nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm([inter_height * inter_width * output_channels]),
            nn.Linear(inter_height * inter_width * output_channels, first_hidden_size),
            nn.ReLU(),
            self.create_hidden_layers(num_hidden_layers, first_hidden_size, 16),
            nn.Linear(16, 1),
        )

    def forward(self, s_s: torch.Tensor):
        cnn_s_s = self.thermo_cnn(s_s)
        thermostability = self.thermo_module_regression(cnn_s_s)
        return thermostability.squeeze(1)

    def create_hidden_layers(
        self, num: int, input_size: int, output_size: int
    ) -> nn.Module:
        if num == 1:
            return nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())

        result = []
        for i in range(num - 1):
            print(input_size)
            _output_size = int(input_size / 2)
            layer = nn.Linear(input_size, _output_size)
            result.append((str(i * 2), layer))
            result.append((str(i * 2 + 1), nn.ReLU()))
            input_size = _output_size

        result.append((str((num - 1) * 2), nn.Linear(input_size, output_size)))
        result.append((str((num - 1) * 2 + 1), nn.ReLU()))

        return nn.Sequential(OrderedDict(result))

class CNNPregeneratedFullHeightFC(nn.Module):
    def __init__(self,  num_hidden_layers=1, first_hidden_size=512):
        super().__init__()

        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])
        output_channels = 16

        self.thermo_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(3, 1024), stride=(1, 1)),
        )
        inter_height = 1
        inter_width = 698

        self.thermo_module_regression = torch.nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm([inter_height * inter_width * output_channels]),
            nn.Linear(inter_height * inter_width * output_channels, first_hidden_size),
            nn.ReLU(),
            self.create_hidden_layers(num_hidden_layers, first_hidden_size, 16),
            nn.Linear(16, 1),
        )

    def forward(self, s_s: torch.Tensor):
        cnn_s_s = self.thermo_cnn(s_s)
        thermostability = self.thermo_module_regression(cnn_s_s)
        return thermostability.squeeze(1)

    def create_hidden_layers(
        self, num: int, input_size: int, output_size: int
    ) -> nn.Module:
        if num == 1:
            return nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())

        result = []
        for i in range(num - 1):
            print(input_size)
            _output_size = int(input_size / 2)
            layer = nn.Linear(input_size, _output_size)
            result.append((str(i * 2), layer))
            result.append((str(i * 2 + 1), nn.ReLU()))
            input_size = _output_size

        result.append((str((num - 1) * 2), nn.Linear(input_size, output_size)))
        result.append((str((num - 1) * 2 + 1), nn.ReLU()))

        return nn.Sequential(OrderedDict(result))
