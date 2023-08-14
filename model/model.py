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


class PatchEmbedding2D(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, emb_size: int):
        super(PatchEmbedding2D, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # [B, emb_size, H', W']


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_size, max_length):
        super(PositionalEmbedding, self).__init__()
        # self.pos_emb = nn.Parameter(torch.zeros(1, max_length, emb_size))
        self.pos_emb = nn.Parameter(torch.randn(1, emb_size))

    def forward(self, x):
        return x + self.pos_emb


class PositionalEmbedding2D(nn.Module):
    def __init__(self, dim: int, h: int, w: int):
        super(PositionalEmbedding2D, self).__init__()
        
        self.row_embed = nn.Embedding(h, dim // 2)
        self.col_embed = nn.Embedding(w, dim // 2)
        
        self.h, self.w = h, w
        self.dim = dim
        
    def forward(self, x):
        # x shape: [B, C, H, W]
        
        # Create positional indices
        rows = torch.arange(0, self.h, device=x.device).unsqueeze(-1)
        cols = torch.arange(0, self.w, device=x.device).unsqueeze(0)
        
        row_embeds = self.row_embed(rows)
        col_embeds = self.col_embed(cols)
        
        # Combine row and column embeddings to create a grid
        pos_embed = torch.cat([row_embeds.expand(self.h, self.w, self.dim // 2), 
                               col_embeds.expand(self.h, self.w, self.dim // 2)], dim=2)  # [H, W, dim]
        
        # Reshape to match the shape of x
        pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], self.dim, self.h, self.w)
        
        return x + pos_embed


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

#         flat_kernel_size = self.kernel_size ** 2
#         flat_stride = self.stride ** 2
#         # for i in range(0, L - self.kernel_size + 1, self.stride):
#         #     x_view = x[:, i:i+self.kernel_size, :]
#         #     attn_out = self.comp_attention(x_view)
#         #     out[:, i:i+self.kernel_size, :] += attn_out
#         for i in range(0, L - flat_kernel_size + 1, flat_stride):
#             x_view = x[:, i:i+flat_kernel_size, :]
#             attn_out = self.comp_attention(x_view)
#             out[:, i:i+flat_kernel_size, :] += attn_out
        
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
    def __init__(self, dim: int, kernel_size: int = 2, stride: int = 1, heads: int = 8, dropout: float = 0.1):
        super(SlidingKernelAttention, self).__init__()
        self.head = heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.scale = (dim // heads) ** -0.5
        
        # Define the QKV projection layer
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        
        # Define the output projection layer
        self.to_out = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)

    def comp_attention(self, x):
        # Calculate Q, K, V values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # Compute the attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        return attn @ v

    def forward(self, x):
        B, C, H, W = x.shape
        out = torch.zeros_like(x)

        # Use unfold to extract sliding windows (2D kernels) from the 2D patch grid
        x_unfold = nn.functional.unfold(x, kernel_size=self.kernel_size, 
                                        stride=self.stride).permute(0, 2, 1).view(B, -1, C, self.kernel_size, self.kernel_size)
        
        # Move the kernel dimensions to the end
        x_unfold = x_unfold.permute(0, 2, 3, 4, 1).reshape(B*C*self.kernel_size*self.kernel_size, -1, self.kernel_size*self.kernel_size)
        
        # Compute attention for each kernel position
        attn_out = self.comp_attention(x_unfold)
        
        # Reshape the attention output to match the original shape
        attn_out = attn_out.view(B, C, self.kernel_size, self.kernel_size, H - self.kernel_size + 1, W - self.kernel_size + 1)
        
        # Aggregate the results to produce the final attention output
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                out[:, :, i:H-self.kernel_size+1+i, j:W-self.kernel_size+1+j] += attn_out[:, :, i, j]
        
        return out


class KernelTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, kernel_size=8, stride=4, mlp_ratio=2, drop=0.1):
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
        self.patch_embed = PatchEmbedding2D(in_channels, patch_size, emb_size)
        # self.num_patches = (emb_size // patch_size) ** 2
        # self.pos_embed = PositionalEmbedding(emb_size, self.num_patches + 1)
        grid_size = 32 // patch_size
        self.pos_embed = PositionalEmbedding2D(emb_size, grid_size, grid_size)
        self.cls_token = nn.Parameter(torch.zeros(1, emb_size, 1, 1))
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
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_token, x), dim=1)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1, x.shape[-1])
        x = torch.cat((cls_token, x), dim=2)
        for blk in self.blocks:
            x = blk(x)
        # x = x.mean(dim=[2,3])  # Global average pooling
        cls_output = x[:, :, 0, 0]
        # return self.classifier(x)
        return self.classifier(cls_output)


# if __name__ == '__main__':
#     model = KernelTransformer(in_channels=3, emb_size=512, patch_size=2, 
#                               num_blocks=6, heads=8, num_classes=10)
#     print(model)
#     print(f"Number of parameters: {count_parameters(model):,}")
