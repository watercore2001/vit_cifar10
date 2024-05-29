from torch import nn
import torch
from einops.layers.torch import Rearrange
from einops import repeat
from .transformer import TransformerArgs, Transformer


class VisionTransformer(nn.Module):
    def __init__(self, img_size: int, in_chans: int, patch_size: int,
                 transformer_arg: TransformerArgs, num_classes: int):
        # 1. convolution
        # self.patch_embed = nn.Conv2d(in_chans=in_chans, transformer_arg.embedding_dim, kernel_size=patch_size, stride=patch_size)
        # 2. linear embedding
        self.patch_embed = nn.Sequential(
            Rearrange(pattern="b c (new_h p1) (new_w p2) -> b (new_h new_w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.Linear(patch_size ** 2 * in_chans, transformer_arg.embedding_dim))

        assert img_size % patch_size == 0
        self.pos_encoding = nn.Parameter(torch.randn(1, (img_size//patch_size)**2, transformer_arg.embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_arg.embedding_dim))

        self.transformer = Transformer(transformer_args=transformer_arg)

        self.linear = nn.Linear(transformer_arg.embedding_dim, num_classes)

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.patch_embed(x)
        # x: b l c
        # self.pos_encoding: 1 l c
        x += self.pos_encoding

        # cls_token: 1 1 c
        cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls_token, x], dim=1)  # x: b 1+l d

        x = self.transformer(x)

        x = x[:, 0, :]  # b c

        x = self.linear(x)  # b num

        return x

