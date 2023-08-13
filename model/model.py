import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import count_parameters


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
    

# class SlidingKernelAttention(nn.Module):
#     def __init__(self, dim, heads=8, kernel_size=4, stride=2):
#         super(SlidingKernelAttention, self).__init__()
#         self.heads = heads
#         self.scale = dim ** -0.5
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.to_out = nn.Linear(dim, dim)

#     def forward(self, x):
#         B, L, C = x.shape
#         out = torch.zeros_like(x)

#         for i in range(0, L - self.kernel_size + 1, self.stride):
#             x_view = x[:, i:i+self.kernel_size, :]
#             attn_out = self.comp_attention(x_view)
#             out[:, i:i+self.kernel_size, :] += attn_out
        
#         return out
    
#     def comp_attention(self, x_view):
#         B, L, C = x_view.shape
#         qkv = self.to_qkv(x_view).chunk(3, dim=-1)
#         q, k, v = map(lambda t: t.reshape(B, L, self.heads, C // self.heads).permute(0, 2, 1, 3), qkv)
        
#         dots = (q @ k.transpose(-1, -2)) * self.scale
#         attn = dots.softmax(dim=-1)
        
#         out = attn @ v
#         out = out.transpose(1, 2).reshape(B, L, C)
#         return self.to_out(out)


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

        def comp_attention(x_view):
            B, L, C = x_view.shape
            qkv = self.to_qkv(x_view).chunk(3, dim=-1)
            q, k, v = map(lambda t: t.reshape(B, L, self.heads, C // self.heads).permute(0, 2, 1, 3), qkv)

            dots = (q @ k.transpose(-1, -2)) * self.scale
            attn = dots.softmax(dim=-1)

            out = attn @ v
            out = out.transpose(1, 2).reshape(B, L, C)
            return self.to_out(out)

        with nn.parallel.parallel_apply([comp_attention] * ((L - self.kernel_size) // self.stride + 1), 
                                        [(x[:, i:i+self.kernel_size, :],) for i in range(0, L - self.kernel_size + 1, self.stride)]) as attn_outs:
            for i, attn_out in enumerate(attn_outs):
                out[:, i*self.stride:i*self.stride+self.kernel_size, :] += attn_out
        
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_size, max_length):
        super(PositionalEmbedding, self).__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, max_length, emb_size))

    def forward(self, x):
        return x + self.pos_emb


class KernelTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, kernel_size=8, stride=4, mlp_ratio=4, drop=0.1):
        super(KernelTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        # self.attention = KernelAttention(dim, heads=heads)
        # self.attention = nn.MultiheadAttention(dim, heads)
        self.attention = SlidingKernelAttention(dim, heads=heads, 
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
        # self.pos_emb = PositionalEmbedding(dim, dim)

    def forward(self, x):
        nor = self.norm1(x)
        attn_out = self.attention(nor)
        # attn_out, _ = self.attention(nor, nor, nor)
        attn_out = self.dropout(attn_out)
        x = x + attn_out

        nor = self.norm2(x)
        mlp_out = self.mlp(nor)
        x = x + mlp_out
        
        # return self.pos_emb(x)
        return x


class KernelTransformer(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, num_blocks, heads, num_classes):
        super(KernelTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size)
        self.pos_embed = PositionalEmbedding(emb_size, emb_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.small_blocks = num_blocks // 3
        self.medium_blocks = num_blocks // 3
        self.large_blocks = num_blocks - self.small_blocks - self.medium_blocks
        # self.blocks = nn.ModuleList(
        #     [KernelTransformerBlock(emb_size, heads) for _ in range(num_blocks)]
        # )
        # self.blocks = nn.ModuleList(
        #     [KernelTransformerBlock(emb_size, heads=4, kernel_size=4, stride=2)] * self.small_blocks +
        #     [KernelTransformerBlock(emb_size, heads=6, kernel_size=8, stride=4)] * self.medium_blocks +
        #     [KernelTransformerBlock(emb_size, heads=8, kernel_size=16, stride=8)] * self.large_blocks
        # )
        self.blocks = nn.ModuleList(
            [KernelTransformerBlock(emb_size, heads=4, kernel_size=4, stride=2),
             KernelTransformerBlock(emb_size, heads=4, kernel_size=4, stride=2),
             KernelTransformerBlock(emb_size, heads=4, kernel_size=4, stride=2),
             KernelTransformerBlock(emb_size, heads=8, kernel_size=8, stride=4),
             KernelTransformerBlock(emb_size, heads=8, kernel_size=8, stride=4),
             KernelTransformerBlock(emb_size, heads=8, kernel_size=8, stride=4),
             KernelTransformerBlock(emb_size, heads=16, kernel_size=16, stride=8),
             KernelTransformerBlock(emb_size, heads=16, kernel_size=16, stride=8)]
        )
        self.classifier = nn.Linear(emb_size, num_classes) # Added classifier head

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)


# if __name__ == '__main__':
#     model = KernelTransformer(in_channels=3, emb_size=256, patch_size=2, 
#                               num_blocks=6, heads=8, num_classes=10)
#     print(model)
#     print(f"Number of parameters: {count_parameters(model):,}")
