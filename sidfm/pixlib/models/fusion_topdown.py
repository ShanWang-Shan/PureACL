"""
Topdown fusion with transformer.
"""

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat

from sidfm.visualization.viz_2d import (plot_images, features_to_RGB, add_text, save_plot)
import matplotlib.pyplot as plt
import matplotlib as mpl

# helpers

# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)

# def posemb_sincos_2d(pe, dim, temperature = 10000):
#     device = pe.device
#     pe = pe.permute(0,2,3,1) #[b,h,w,2]
#     assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
#     omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
#     omega = 1. / (temperature ** omega)
#
#     pe = pe[..., None] * omega
#     pe = torch.cat((pe.sin(), pe.cos()), dim = -1)
#     return pe.flatten(-2)

class Fusion_topdown(nn.Module):
    def __init__(self, dim, dim_head=64, bias=False, norm=nn.LayerNorm):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, 1, bias=False)

        #self.dim_head = dim_head
        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, dim_head, bias=bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, dim_head, bias=bias))

    def add_extra_embed(self, dim):
        layer = self.embed
        # Creating new Conv2d layer
        new_layer = nn.Conv2d(3, dim, 1, bias=False).to(layer.weight)
        new_weight = torch.cat([layer.weight.clone(), new_layer.weight[:,2:].clone()], dim=1)
        new_layer.weight = nn.Parameter(new_weight)
        self.embed = new_layer

    def forward(self, x, pe, v_start, vis=False, img=None):
        # x: [b,c,h,w] 2D features
        # pe: [b,h,w,2] pose embedding
        # v_start: int

        b, c, h, w = x.shape
        #pe = posemb_sincos_2d(pe, c) #[b,2,h,w] to [b,c,h,w]
        pe = self.embed(pe) #[b,c,h,w]
        x_pe = x+pe

        v = x #[b,c,h,w]
        q = rearrange(x_pe[:,:,v_start:], 'b c q w -> (b w) q c')
        k = rearrange(x_pe, 'b c h w -> (b w) h c')

        # Project with multiple heads
        q = self.to_q(q)                                # (b w) q d
        k = self.to_k(k)                                # (b w) h d

        # Dot product attention along h
        dot = torch.einsum('b q d, b h d -> b q h', q, k) # (b w) q h

        # remove attention under query v
        valid = torch.tril(torch.ones_like(dot[0]), diagonal=v_start)
        dot = dot*valid + (1-valid)*-1E9
        att = dot.softmax(dim=-1)

        att = rearrange(att, '(b w) q h -> b w q h', b=b, w=w)

        # visualize att of middle q
        if vis:
            vis_q = (h-v_start)//2
            att_img = att[0,:,vis_q].T.unsqueeze(-1).cpu().detach()
            plot_images([att_img], cmaps=mpl.cm.gnuplot2, dpi=50)
            if img is not None:
                ori_img = img[0].permute(1, 2, 0).cpu().detach()
                axes = plt.gcf().axes
                axes[0].imshow(ori_img, alpha=0.2, extent=axes[0].images[0]._extent)
            plt.show()

        # Combine values (image level features).
        z = torch.einsum('b w q h, b c h w -> b c q w', att, v)

        return z #[b,c,h-v_start,w]

if __name__ == '__main__':

    fusion = Fusion_topdown(
        dim = 64,
        depth = 3,
        heads = 8,
        mlp_dim = 64,
        dropout = 0.1
    )

    h = 128
    w = 256
    b = 2
    input = torch.randn(b, 64, h, w)
    v = torch.arange(0, h)/ h * 2 - 1  # -1~1
    u = torch.arange(0, w) / w * 2 - 1  # -1~1
    vv, uu = torch.meshgrid(v, u)
    pe = torch.stack([uu, vv], dim=0)  # shape = [2, h, w]
    pe = pe[None, :, :, :].repeat(b, 1, 1, 1)
    logits = fusion(input, pe, int(h*0.65)) #[b,v,h-v_start,w]

