import torch
import torch.nn


def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=None):
        self.margin = margin
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, nagative, normalize_feature=False):
        if normalize_feature:
            anchor = normalize(anchor, axis=-1)
            positive = normalize(positive, axis=-1)
            nagative = normalize(nagative, axis=-1)
        loss = self.triplet_loss(anchor, positive, nagative)
        return loss
