# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from typing import Tuple
from timm.models.vision_transformer import Block, PatchEmbed
from models.utils import get_2d_sincos_pos_embed_flexible
from functools import partial


class MaskedAutoEncoderViT(nn.Module):
    def __init__(self, input_size: Tuple[int, int], patch_size: int, channels: int,
                 embed_dim: int, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int):
        super().__init__()
        self.input_size = input_size
        self.channels = channels
        self.mlp_ratio = 4.

        # Masked Autoencoder (MAE) Encoder
        self.patch_embed = PatchEmbed(img_size=input_size, patch_size=patch_size,
                                      in_chans=self.channels, embed_dim=embed_dim,
                                      norm_layer=partial(nn.LayerNorm, eps=1e-6))

        num_patches = self.patch_embed.num_patches
        self.grid_h = int(input_size[0] // patch_size)
        self.grid_w = int(input_size[1] // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.encoder_block = nn.ModuleList([
            Block(embed_dim, encoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(encoder_depths)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Masked Autoencoder (MAE) Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
        self.decoder_block = nn.ModuleList([
             Block(decoder_embed_dim, decoder_heads, self.mlp_ratio, qkv_bias=True,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6))
             for _ in range(decoder_depths)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * self.channels, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1],
                                                     (self.grid_h, self.grid_w),
                                                     cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1],
                                                             (self.grid_h, self.grid_w),
                                                             cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def random_masking(x, mask_ratio):
        n, l, d = x.shape  # batch, length, dim
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.encoder_block:
            x = block(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore: torch.Tensor):
        # embed tokens
        x = self.decoder_embed(x[:, 1:, :])

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for block in self.decoder_block:
            x = block(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        return x

    def patchify(self, image):
        p = self.patch_embed.patch_size[0]
        h = image.shape[2] // p
        w = image.shape[3] // p

        x = image.reshape(shape=(image.shape[0], self.channels, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(image.shape[0], h * w, p ** 2 * self.channels))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = self.input_size[0] // p
        w = self.input_size[1] // p

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        image = x.reshape(shape=(x.shape[0], self.channels, h * p, w * p))
        return image

    def forward(self, image, mask_ratio=0.8):
        real = self.patchify(image.clone())
        latent, mask, ids_restore = self.forward_encoder(image, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return real, pred, mask, latent


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
