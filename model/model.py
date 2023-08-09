import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class KernelAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super(KernelAttention, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.fc(out)


class KernelTransformerBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, stride=1, padding=2):
        super(KernelTransformerBlock, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.attention = KernelAttention(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Extract patches (our "windows") from the image
        windows = F.unfold(x, self.kernel_size, stride=self.stride, padding=self.padding)
        windows = windows.permute(0, 2, 1).reshape(B, -1, C)

        # Apply the attention to each window
        attended_windows = self.attention(windows)

        # Here, we might want to fold the windows back to the spatial form. 
        # For simplicity, this is not done in this example, but in a real-world scenario, you'd fold it back.

        return self.mlp(attended_windows)


class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PatchMerging, self).__init__()
        self.merge = nn.Linear(in_dim * 2 * 2, out_dim)

    def forward(self, x):
        B, L, D = x.shape
        H = W = int(sqrt(L))
        x = x.view(B, H, W, D).permute(0, 3, 1, 2).reshape(B, D, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, H // 2, W // 2, D * 4).permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
        return self.merge(x)


class HierarchicalLayers(nn.Module):
    def __init__(self, embed_dim, window_size, heads, depth, num_blocks):
        super(HierarchicalLayers, self).__init__()

        layers = []
        for _ in range(depth):
            for _ in range(num_blocks):
                layers.append(KernelTransformerBlock(dim=embed_dim, window_size=window_size, heads=heads))
            if _ < depth - 1:
                layers.append(PatchMerging(embed_dim, embed_dim * 2))
                embed_dim *= 2

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class HierarchicalKernelTransformer(nn.Module):
    def __init__(self, in_channels, img_dim, embed_dim, window_size, heads, num_classes=10, depth=4, num_blocks=2):
        super(HierarchicalKernelTransformer, self).__init__()

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size=4)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_dim // 4)**2, embed_dim))  # for an initial patch size of 4
        
        self.hierarchical_layers = HierarchicalLayers(embed_dim, window_size, heads, depth, num_blocks)
        
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x += self.pos_embed
        x = self.hierarchical_layers(x)
        x = x.mean(dim=1)
        return self.fc(x)
