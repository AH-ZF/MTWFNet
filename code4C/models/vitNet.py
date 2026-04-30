'''
Complete VIT module

'''

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor
import torch.nn.functional as F
import os, random
import torch
from torch import nn
import numpy as np


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return x


class FeedForward(nn.Module):  # MLP
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.net2 = nn.Sequential(
            # nn.Linear(dim, hidden_dim),
            # nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x = self.net(x)
        x = self.net2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size // num_heads
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.con4d = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x1 = torch.cat((torch.unsqueeze(x[:, 0, :], dim=1), torch.unsqueeze(x[:, 2, :], dim=1)), dim=1)
        x2 = torch.cat((torch.unsqueeze(x[:, 1, :], dim=1), torch.unsqueeze(x[:, 3, :], dim=1)), dim=1)
        qkv1 = rearrange(self.qkv(x1), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        qkv2 = rearrange(self.qkv(x2), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        #
        queries1, keys1, values1 = qkv1[0], qkv1[1], qkv1[2]
        queries2, keys2, values2 = qkv2[0], qkv2[1], qkv2[2]
        # queries, keys, values = queries2, keys2, values1
        # queries = torch.cat((queries1, queries2), dim=2)
        # keys = torch.cat((keys1, keys2), dim=2)
        # values = values1+values2
        # qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        # queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy1 = torch.einsum('bhqd, bhkd -> bhqk', queries1, keys1)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy1.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att1 = F.softmax(energy1, dim=-1) / scaling
        att1 = self.att_drop(att1)

        energy2 = torch.einsum('bhqd, bhkd -> bhqk', queries2, keys2)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy2.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att2 = F.softmax(energy2, dim=-1) / scaling
        att2 = self.att_drop(att2)
        # Method 1
        out11 = torch.einsum('bhal, bhlv -> bhav ', att1, values1)
        out11 = rearrange(out11, "b h n d -> b n (h d)")
        out11 = self.projection(out11)

        out12 = torch.einsum('bhal, bhlv -> bhav ', att1, values2)
        out12 = rearrange(out12, "b h n d -> b n (h d)")
        out12 = self.projection(out12)

        out21 = torch.einsum('bhal, bhlv -> bhav ', att2, values1)
        out21 = rearrange(out21, "b h n d -> b n (h d)")
        out21 = self.projection(out21)

        out22 = torch.einsum('bhal, bhlv -> bhav ', att2, values2)
        out22 = rearrange(out22, "b h n d -> b n (h d)")
        out22 = self.projection(out22)
        # Method 2
        # BBB=att1+att2
        # AAA=F.sigmoid(att1+att2)
        # out1 = torch.einsum('bhal, bhlv -> bhav ', 0.5*(att1+att2), values1)
        # out2 = torch.einsum('bhal, bhlv -> bhav ', 0.5*(att1+att2), values2)
        # out1 = rearrange(out1, "b h n d -> b n (h d)")
        # out1 = self.projection(out1)
        # out2 = rearrange(out2, "b h n d -> b n (h d)")
        # out2 = self.projection(out2)

        out = torch.cat((out11 + out12, out21 + out22), dim=1)
        # out = self.con4d(out)
        # out = torch.cat((out1, out2), dim=1)

        # sum up over the third axis
        # out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # out = rearrange(out, "b h n d -> b n (h d)")
        # out = self.projection(out)
        return out, [att1, att2]


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.):
        super().__init__()
        # inner_dim = dim_head * heads
        dim_head = dim // heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]796/8
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        # out = rearrange(out, 'b h n d -&gt; b n (h d)')
        out = self.to_out(out)
        return out, [q, k, v]


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        # self.vecdim = dim
        self.vecdim = 128
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(self.vecdim, MultiHeadAttention(emb_size=self.vecdim, num_heads=heads, dropout=dropout)),
                PreNorm(self.vecdim, FeedForward(self.vecdim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # x[1] = attn(x[1]) + x[1]
            # x[1] = ff(x[1]) + x[1]
            # x[0] = attn(x[0]) + x[0]
            # x[0] = ff(x[0]) + x[0]

            x = attn(x)[0] + x
            x = ff(x) + x
        return x, attn(x)[1]


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        # self.emb_size = self.patch_size * self.patch_size * in_channels
        self.emb_size = emb_size
        # self.projection = nn.Sequential(
        #     # using a conv layer instead of a linear one -> performance gains
        #     nn.Conv2d(in_channels, self.emb_size, kernel_size=self.patch_size, stride=self.patch_size),
        #     Rearrange('b e (h) (w) -> b (h w) e'),
        # )
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv1d(in_channels, emb_size, kernel_size=emb_size, stride=emb_size),
            # Rearrange('b e (h) (w) -> b (h w) e'),
        )

        # self.cls_token = nn.Parameter(torch.randn(1, 2, emb_size))
        self.cls_token = nn.Parameter(torch.randn(1, 2, 128))
        # self.positions = nn.Parameter(torch.randn((img_size // self.patch_size) ** 2 + 1, self.emb_size))
        # self.positions = nn.Parameter(torch.randn(patch_size + 2, emb_size))
        # self.positions = nn.Parameter(torch.randn(patch_size + 2, 128))
        self.positions = nn.Parameter(torch.randn(2 + 2, 128))

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        b, _, _ = x.shape
        # x = self.projection(x)
        # x = rearrange(x, 'b c l -> b l c')
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions

        return x


class ViT(nn.Module):
    def __init__(self, *, vec_size, patch_size, patchvecL, depth, heads, mlp_dim, pool='cls', channels=1,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        # image_height, image_width = pair(image_size)  ## 224*224
        patch_height, patch_width = pair(patch_size)  ## 16 * 16
        print(f'patch_height={patch_height}')
        print(f'patch_width={patch_width}')

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # patch
        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch
        patch_dim = vec_size // patch_size
        print(f'patch_dim={patch_dim}')
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.positionEmb = PatchEmbedding(in_channels=channels, patch_size=patch_size, emb_size=patch_dim,
                                          img_size=vec_size)
        self.transformer = Transformer(patchvecL, depth, heads, mlp_dim, dropout)

        pass

    def forward(self, img):
        # rgbimg = torch.unsqueeze(img[:, 0, :], dim=1)
        # ecgimg = torch.unsqueeze(img[:, 1, :], dim=1)
        # rgbx = self.positionEmb(rgbimg)
        # ecgx = self.positionEmb(ecgimg)
        # x = torch.cat([rgbx, ecgx], dim=1)
        # x = torch.cat([rgbimg, ecgimg], dim=1)
        # x = self.transformer(x)
        img = self.positionEmb(img)
        x, ww = self.transformer(img)
        # rgbx = self.transformer(rgbx)
        # coffw = torch.cat([ww[0].view(ww[0].size(0), -1), ww[1].view(ww[1].size(0), -1)], dim=1)

        coffw = [ww[0].view(ww[0].size(0), -1), ww[1].view(ww[1].size(0), -1)]
        # x = ecgx + rgbx
        # x = self.positionEmb(img)
        # x = self.transformer(x)
        return x, coffw
