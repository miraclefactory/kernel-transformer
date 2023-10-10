import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, emb_size: int):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, emb_size, H', W']
        return x.flatten(2).transpose(1, 2)  # [B, num_patches, emb_size]


class PatchEmbedding2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, permute: bool = True):
        super(PatchEmbedding2D, self).__init__()
        self.patch_size = patch_size
        self.permute = permute
        self.proj = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.linear = nn.Linear(in_channels * patch_size ** 2, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.permute:
            B, H, W, C = x.shape
            x = x.permute(0, 3, 1, 2)
        else:
            B, C, H, W = x.shape
        H, W = H // self.patch_size, W // self.patch_size
        x = self.proj(x).view(B, -1, H, W).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_size, max_length):
        super(PositionalEmbedding, self).__init__()
        # self.pos_emb = nn.Parameter(torch.zeros(1, emb_size))
        self.pos_emb = nn.Parameter(torch.randn(1, emb_size))

    def forward(self, x):
        return x + self.pos_emb


class SlidingKernelAttention(nn.Module):
    def __init__(self, dim, heads=8, kernel_size=4, stride=2):
        super(SlidingKernelAttention, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.kernel_size = kernel_size
        self.stride = stride
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        out = torch.zeros_like(x)

        for i in range(0, L - self.kernel_size + 1, self.stride):
            x_view = x[:, i:i+self.kernel_size, :]
            attn_out = self.comp_attention(x_view)
            out[:, i:i+self.kernel_size, :] += attn_out

        return out

    def comp_attention(self, x_view):
        B, L, C = x_view.shape
        qkv = self.to_qkv(x_view).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, L, self.heads, C // self.heads).permute(0, 2, 1, 3), qkv)

        dots = (q @ k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.to_out(out)


class SlidingKernelAttention2D(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 2, stride: int = 1, heads: int = 8):
        super(SlidingKernelAttention2D, self).__init__()
        self.heads = heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.scale = (dim // heads) ** -0.5

        # Relative positional bias
        self.rel_embed_h = nn.Parameter(torch.randn(kernel_size, dim // heads))
        self.rel_embed_w = nn.Parameter(torch.randn(kernel_size, dim // heads))

        # QKV projection layer
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        # Output projection layer
        self.to_out = nn.Linear(dim, dim)

    def comp_attention(self, x_view):
        B, L, C = x_view.shape
        qkv = self.to_qkv(x_view).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, L, self.heads, C // self.heads).permute(0, 2, 1, 3), qkv)

        dots = (q @ k.transpose(-1, -2)) * self.scale

        h_bias = self.relative_positional_bias(q, self.rel_embed_h)
        w_bias = self.relative_positional_bias(q, self.rel_embed_w)
        dots = dots + h_bias + w_bias

        attn = dots.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.to_out(out)

    def relative_positional_bias(self, q, rel_embed):
        B, H, L, C = q.shape
        scores = q @ rel_embed.transpose(0, 1)
        scores = scores.reshape(B, H, L, 1, self.kernel_size).expand(-1, -1, -1, L, -1)
        scores = scores.sum(dim=-1)
        return scores

    def forward(self, x):
        B, H, W, C = x.shape
        out = torch.zeros_like(x)

        for i in range(0, H - self.kernel_size + 1, self.stride):
            for j in range(0, W - self.kernel_size + 1, self.stride):
                x_window = x[:, i:i+self.kernel_size, j:j+self.kernel_size, :]
                # Reshape for attention
                x_view = x_window.permute(0, 3, 1, 2).reshape(B, -1, C)
                attn_out = self.comp_attention(x_view)
                # Reshape back to spatial format
                attn_out = attn_out.reshape(B, C, self.kernel_size, self.kernel_size).permute(0, 2, 3, 1)
                out[:, i:i+self.kernel_size, j:j+self.kernel_size, :] += attn_out

        return out


class KernelTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, kernel_size=8, stride=4, mlp_ratio=4, drop=0.1):
        super(KernelTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        # self.attention = KernelAttention(dim, heads=heads)
        # self.attention = nn.MultiheadAttention(dim, heads)
        self.attention = SlidingKernelAttention2D(dim, heads=heads, 
                                                  kernel_size=kernel_size, stride=stride)
        self.dropout = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        nor = self.norm1(x)
        attn_out = self.attention(nor)
        attn_out = self.dropout(attn_out)
        x = x + attn_out

        nor = self.norm2(x)
        mlp_out = self.mlp(nor)
        x = x + mlp_out

        return x


class KernelTransformerStage(nn.Module):
    def __init__(self, dim, num_blocks, heads=8, kernel_size=8, stride=4, mlp_ratio=4, drop=0.1):
        super(KernelTransformerStage, self).__init__()
        self.blocks = nn.ModuleList([
            KernelTransformerBlock(dim, heads, kernel_size, stride, mlp_ratio, drop)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class KernelTransformer(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, heads, num_classes, struct):
        super(KernelTransformer, self).__init__()
        # self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size)
        # self.num_patches = (emb_size // patch_size) ** 2
        # self.pos_embed = PositionalEmbedding(emb_size, self.num_patches + 1)
        # grid_size = 32 // patch_size
        # self.pos_embed = PositionalEmbedding2D(emb_size, grid_size, grid_size)
        # self.cls_token = nn.Parameter(torch.zeros(1, emb_size, 1, 1))
        self.blocks = nn.ModuleList([
            PatchEmbedding2D(in_channels, emb_size, 2, permute=False),
            KernelTransformerStage(emb_size, struct[0], heads=4, kernel_size=4, stride=2),
            PatchEmbedding2D(emb_size, emb_size * 2, 2),
            KernelTransformerStage(emb_size * 2, struct[1], heads=8, kernel_size=4, stride=2),
            PatchEmbedding2D(emb_size * 2, emb_size * 4, 2),
            KernelTransformerStage(emb_size * 4, struct[2], heads=16, kernel_size=4, stride=2),
            PatchEmbedding2D(emb_size * 4, emb_size * 8, 1),
            KernelTransformerStage(emb_size * 8, struct[3], heads=32, kernel_size=4, stride=2)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size * 8),
            nn.Linear(emb_size * 8, num_classes)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=[1,2])  # Global average pooling
        return self.classifier(x)


if __name__ == '__main__':
    from utils import count_parameters

    model = KernelTransformer(in_channels=3, emb_size=96, patch_size=2, 
                              heads=8, num_classes=10, struct=(2, 2, 6, 2))
    print(model) # print model architecture
    print(f"Number of parameters: {count_parameters(model):,}")
