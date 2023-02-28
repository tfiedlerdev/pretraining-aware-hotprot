from torch import nn
import torch
from thermostability.hotinfer_pregenerated import create_fc_layers


class RepresentationSummarizerSingleInstance(nn.Module):
    def __init__(self, num_hidden_layers=1, per_residue_output_size=1):
        super().__init__()

        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])

        # todo: experiment for 1st summarizer with input 1024, output e.g. 10
        # followed by FC network with input size 700x10
        self.num_hidden_layers = num_hidden_layers
        self.per_residue_output_size = per_residue_output_size
        self.summarizer = create_fc_layers(
            num_hidden_layers, 1024, per_residue_output_size, p_dropout=0
        )

    def forward(self, s_s: torch.Tensor):
        # [-1, sequence_len, 1024]
        # [sequence_len, -1, 1024]
        per_residue = s_s.view((700, -1, 1024))
        summaries = []
        for residue_repr_batch in per_residue:
            residue_summary = self.summarizer(residue_repr_batch)
            summaries.append(residue_summary)

        stacked = torch.stack(summaries, dim=1)
        return stacked


class RepresentationSummarizer700Instance(nn.Module):
    def __init__(self, num_hidden_layers=0, per_residue_output_size=1):
        super().__init__()

        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])

        self.num_hidden_layers = num_hidden_layers
        self.per_residue_output_size = per_residue_output_size
        self.summarizers = [
            create_fc_layers(
                num_hidden_layers, 1024, per_residue_output_size, p_dropout=0
            ).to("cuda:0")
            for _ in range(700)
        ]

    def forward(self, s_s: torch.Tensor):
        # [-1, sequence_len, 1024]
        # [sequence_len, -1, 1024]
        per_residue = s_s.view((700, -1, 1024))
        summaries = []
        for i, residue_repr_batch in enumerate(per_residue):
            residue_summary = self.summarizers[i](residue_repr_batch)
            summaries.append(residue_summary)

        stacked = torch.stack(summaries, dim=1)
        return stacked
