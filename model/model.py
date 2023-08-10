import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import count_parameters


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, emb_size: int):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, emb_size, H', W']
        return x.flatten(2).transpose(1, 2)  # [B, num_patches, emb_size]


class KernelAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super(KernelAttention, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, L, self.heads, C // self.heads).permute(0, 2, 1, 3), qkv)
        
        dots = (q @ k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.to_out(out)


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_size, max_length):
        super(PositionalEmbedding, self).__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, max_length, emb_size))

    def forward(self, x):
        return x + self.pos_emb


class KernelTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super(KernelTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        # self.attention = KernelAttention(dim, heads=heads)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
        self.pos_emb = PositionalEmbedding(dim, dim)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + attn_out  # Residual connection here

        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out  # Residual connection here
        
        return self.pos_emb(x) 


class KernelTransformer(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, num_blocks, heads, num_classes):
        super(KernelTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size)
        # self.pos_embed = PositionalEmbedding(emb_size, emb_size)
        self.blocks = nn.ModuleList([KernelTransformerBlock(emb_size, heads) for _ in range(num_blocks)])
        self.classifier = nn.Linear(emb_size, num_classes) # Added classifier head

    def forward(self, x):
        x = self.patch_embed(x)
        # x = self.pos_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)


if __name__ == '__main__':
    model = KernelTransformer(in_channels=3, emb_size=256, patch_size=2, 
                              num_blocks=12, heads=8, num_classes=10)
    print(model)
    print(f"Number of parameters: {count_parameters(model):,}")
