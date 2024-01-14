# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from typing import List
from models.utils import LinearLayer


class BarlowTwins(nn.Module):
    def __init__(self, backbone, projection_hidden: List, projection_size: int, lambd: float = 0.005):
        super().__init__()
        self.backbone = backbone
        self.projection_hidden = [self.backbone.final_length] + projection_hidden + [projection_size]
        self.projection = nn.Sequential(
            *[
                nn.Sequential(LinearLayer(in_features=i_, out_features=o_, use_bias=True, use_bn=True), nn.ELU())
                for i_, o_ in zip(self.projection_hidden[:-1], self.projection_hidden[1:])]
        )
        self.bn = nn.BatchNorm1d(projection_size, affine=False)
        self.lambd = lambd

    def forward(self, x):
        x1, x2 = x
        batch_size = x[0].shape[0]
        z1 = self.projection(self.backbone(x1))
        z2 = self.projection(self.backbone(x2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


if __name__ == '__main__':
    from models.backbone import CNNEncoder2D
    backbone_ = CNNEncoder2D(sampling_rate=125)
    BarlowTwins(
        backbone=backbone_,
        projection_hidden=[4048, 4048],
        projection_size=1024
    )