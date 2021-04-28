import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, weight=None, gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


def get_cbf_loss(class_counts, beta=0.9999):
    counts = class_counts
    for i in range(len(counts)):
        counts[i] = max(counts[i], 1)

    def cbf_loss(x, y):
        criterion = FocalLoss(reduction='none')
        loss = criterion(x, y)
        cb = torch.tensor(counts).to(y)
        cb = (1 - beta) / (1 - beta ** cb)
        cb = len(counts) * cb / cb.sum()
        cb = torch.gather(cb, 0, y)
        return torch.mean(cb * loss)

    return cbf_loss
