import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Reduced downsampling for 48x48 input
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2)  # 48->24
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GELU(),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        x1 = self.down1(x)  # 32x24x24
        x2 = self.down2(x1) # 64x24x24
        x3 = self.down3(x2) # 128x24x24
        x4 = self.down4(x3) # 256x24x24
        return x4, x3, x2, x1

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Keep original decoder layers
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GELU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64 + 32, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU()
        )
        self.final = nn.Sequential(
            nn.Upsample(size=128, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, skips):
        x3, x2, x1 = skips
        
        # First upsampling (24->48)
        x = self.up1(torch.cat([x, x3], 1))  # 256+128=384 -> 128x48x48
        
        # Upsample x2 (24->48) and process
        x2_up = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, x2_up], 1))  # 128+64=192 -> 64x96x96
        
        # Upsample x1 (24->96) and process
        x1_up = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, x1_up], 1))  # 64+32=96 -> 32x192x192
        
        return self.final(x)  # 1x128x128


class AFNOBlock(nn.Module):
    def __init__(self, dim=256, num_heads=16, mlp_dim=3072, patch_size=1):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        
        # 1. Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,  # Maintain same dimension
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 2. Positional Encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, dim, 24//patch_size, 24//patch_size) * 0.02
        )
        
        # 3. Fourier Attention Components
        self.norm1 = nn.LayerNorm(2*dim)
        self.attn = nn.MultiheadAttention(2*dim, num_heads, batch_first=True)
        
        # 4. MLP Components
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
            nn.GELU()
        )

   def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        # 1. Patch Embedding
        x_patched = self.patch_embed(x)  # [B, C, H/p, W/p]
        x_patched = x_patched + self.pos_embed
        
        # 2. Fourier Transform
        x_fft = torch.fft.rfft2(x_patched, norm='ortho')
        x_combined = torch.cat([x_fft.real, x_fft.imag], dim=1)
        
        # 3. Tokenization
        x_tokens = x_combined.permute(0, 2, 3, 1).view(B, -1, 2*self.dim)
        
        # 4. Self-Attention
        x_tokens = self.norm1(x_tokens)
        attn_out, _ = self.attn(x_tokens, x_tokens, x_tokens)
        attn_out = attn_out.view(B, H//self.patch_size, W//self.patch_size//2+1, 2*self.dim).permute(0, 3, 1, 2)
        
        # 5. Inverse Transform
        x_real, x_imag = torch.chunk(attn_out, 2, dim=1)
        x_ifft = torch.fft.irfft2(torch.complex(x_real, x_imag), 
                                 s=(H//self.patch_size, W//self.patch_size), 
                                 norm='ortho')
        
        # 6. Prepare for folding
        B, C, H_p, W_p = x_ifft.shape
        x_ifft = x_ifft.view(B, C * (self.patch_size**2), -1)
        
        # 7. Reverse Patching
        x = F.fold(
            x_ifft,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        x = x + residual
        
        # 8. MLP
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        
        return x


class SuperResUAFNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.bottleneck = nn.Sequential(*[
            AFNOBlock(
                dim=256,
                num_heads=16,
                mlp_dim=3072,
                patch_size=1  # Can adjust patch size (1=position-wise)
            ) for _ in range(12)
        ])
        self.decoder = Decoder()
        

    def forward(self, x):
        x4, x3, x2, x1 = self.encoder(x)
        x = self.bottleneck(x4)
        return self.decoder(x, (x3, x2, x1))

# Verify dimensions
model = SuperResUAFNO()
x = torch.randn(1, 1, 48, 48)
output = model(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)  # Should be (1, 1, 128, 128)
print(x)
print(output)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")