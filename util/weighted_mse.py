from torch.nn.modules.loss import _Loss
from torch import Tensor, mean
from scipy.stats import norm
from math import sqrt
import torch


class Weighted_MSE_Loss(_Loss):
    def __init__(self, mean: float, var: float, reduction: str = "mean") -> None:
        super(Weighted_MSE_Loss, self).__init__(reduction)
        self.mean = mean
        self.std = sqrt(var)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ds_norm = norm(self.mean, self.std)
        diffs = input - target
        squares = diffs * diffs
        ones = torch.ones_like(target).to("cuda:0")
        pdfs = Tensor(ds_norm.pdf(target.cpu())).to("cuda:0")
        prob = torch.minimum(pdfs, ones)
        weights = 0.5 * (1.0 - prob) + 0.5
        weight_loss = squares * weights
        return mean(weight_loss)
