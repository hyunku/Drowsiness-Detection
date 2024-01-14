# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from typing import Tuple
from timm.models.vision_transformer import PatchEmbed


class EncoderWrapper(nn.Module):
    def __init__(self, input_size: Tuple,
                 patch_embed: PatchEmbed, encoder_block: nn.ModuleList, embed_dim: int,
                 cls_token: nn.Parameter, pos_embed: nn.Parameter,
                 device: torch.device):
        super().__init__()
        self.input_size = input_size

        self.patch_embed = patch_embed
        self.encoder_block = encoder_block
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.cls_token = cls_token
        self.pos_embed = pos_embed
        self.embed_dim = embed_dim
        self.final_length = embed_dim

        self.device = device

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.encoder_block:
            x = block(x)

        x = self.encoder_norm(x)
        x = x[:, 0, :]
        return x
