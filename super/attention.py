import itertools as it
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from pydlr import dlr
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import trange

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, bond_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.bond_dim = bond_dim
        self.d_k = bond_dim // num_heads

        self.q_linear = nn.Linear(input_dim, bond_dim, bias=False)
        self.k_linear = nn.Linear(input_dim, bond_dim, bias=False)
        self.v_linear = nn.Linear(input_dim, bond_dim, bias=False)
        self.out = nn.Linear(bond_dim, input_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Perform linear operation and split into num_heads
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k)

        # Transpose for attention score computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, mask, self.d_k)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.bond_dim)
        output = self.out(concat)

        return output

    def attention(self, q, k, v, mask, d_k):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, bond_dim, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(input_dim, bond_dim, num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        src2 = self.self_attn(src, src, src, mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src

class TransformerEncoderViT(nn.Module):  
    def __init__(self, input_dim, num_heads, num_layers, bond_dim, output_dim, num_patches, dropout=0.0):
        super().__init__()  
        self.layers = nn.ModuleList([EncoderLayer(input_dim, num_heads, bond_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(input_dim)
        self.fc_out = nn.Linear(input_dim, output_dim, bias=False)  # Final output for regression

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        src = self.norm(src)
        
        # For regression, aggregate the patch embeddings 
    
        src = torch.mean(src, dim=1)  # Average across patches

        output = self.fc_out(src)  # Output for regression
        return output
    
class PatchEmbedding(nn.Module):
    def __init__(self, img_height, img_width, patch_size_h, patch_size_w, input_dim, bond_dim):
        super(PatchEmbedding, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        
        # Calculate the number of patches along height and width
        self.num_patches_h = img_height // patch_size_h
        self.num_patches_w = img_width // patch_size_w
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Flatten the patch and apply linear projection
        self.flatten_dim = patch_size_h * patch_size_w * input_dim
        self.linear_proj = nn.Linear(self.flatten_dim, bond_dim)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape and divide into patches
        x = x.view(batch_size, self.img_height, self.img_width, -1)
        x = x.unfold(1, self.patch_size_h, self.patch_size_h).unfold(2, self.patch_size_w, self.patch_size_w)
        
        # Flatten patches and apply linear projection
        x = x.contiguous().view(batch_size, -1, self.flatten_dim)
        x = self.linear_proj(x)
        return x


    
# Vision Transformer: Combining Patch Embedding and Transformer
class VisionTransformer(nn.Module):
    def __init__(self, img_height, img_width, patch_size_h, patch_size_w, input_dim, num_heads, num_layers, bond_dim, output_dim, dropout=0.0):
        super(VisionTransformer, self).__init__()
        
        self.patch_embed = PatchEmbedding(img_height, img_width, patch_size_h, patch_size_w, input_dim, bond_dim)
        
        # Number of patches
        num_patches = (img_height // patch_size_h) * (img_width // patch_size_w)
        
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, bond_dim))  

        # Transformer encoder
        self.transformer = TransformerEncoderViT(bond_dim, num_heads, num_layers, bond_dim, output_dim, num_patches, dropout)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x) 
        
        # Add positional encoding to patch embeddings
        x = x + self.positional_encoding  
        
        # Transformer encoder
        x = self.transformer(x)
        return x

