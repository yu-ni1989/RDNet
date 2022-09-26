#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


def AdpWeights(classes, y):
    ##############
    # Optional. It is not used in default.
    # When it is used, the convergence speed is faster.
    ##############

    y_temp = y.view(y.shape[0], y.shape[1] * y.shape[2])
    counts = torch.zeros(classes)
    for i in range(0, y_temp.shape[0]):
        uni, count = torch.unique(y_temp[i, :], dim=0, return_counts=True)
        for j in range(uni.shape[0]):
            if uni[j] >= classes:
                continue
            counts[uni[j]] = counts[uni[j]] + count[j]

    count_nz = counts.nonzero()
    count_min = counts[count_nz].min()
    # count_mean = counts.mean()
    # counts = counts + count_mean
    counts = counts + count_min
    count_max = counts.max()
    weights = count_max / counts - 1.0
    weights = weights - weights.mean()
    weights = 10.0 / (1.0 + torch.exp(-weights))
    weights = weights - weights.min() + 1.0
    weights = weights.to(y.device)

    return weights


class SelectedLoss(nn.Module):

    def __init__(self, thresh, ignore_lb=255, classes=2, y=None):
        super(SelectedLoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
        if y != None:
            self.criteria = nn.CrossEntropyLoss(weight=AdpWeights(classes, y), ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class SelectedLoss_CPU(nn.Module):

    def __init__(self, thresh, ignore_lb=255, weights=None, y=None):
        super(SelectedLoss_CPU, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float))
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_lb, reduction='none')
        if y != None:
            self.criteria = nn.CrossEntropyLoss(weight=AdpWeights(y), ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


if __name__ == '__main__':
    pass

