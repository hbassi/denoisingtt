import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import trange
import torch.nn.functional as F
# import testUAFNO
# from testUAFNO import SuperResUAFNO

# Set seed and high precision
torch.set_float32_matmul_precision('high')
torch.manual_seed(seed=999)

# ----------------------------
# Data Loading and Preparation
# ----------------------------
def load_data():
    # Load and preprocess data (modify paths as needed)
    input_CG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_coarse_scale_dynamics_1500_tmax=5_sigma=7_numterms=20.npy').reshape((1447, 50, 1, 48, 48))[:100, 0, :, :, :]
    target_FG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_fine_scale_dynamics_1500_tmax=5_sigma=7_numterms=20.npy').reshape((1447, 50, 1, 128, 128))[:100, 0, :, :, :]
    #input_CG = np.load('/pscratch/sd/h/hbassi/2d_vlasov_data_coarse_scale_64.npy')
    #target_FG = np.load('/pscratch/sd/h/hbassi/2d_vlasov_data_coarse_scale_128.npy') 
    # Convert to PyTorch tensors
    input_tensor = torch.tensor(input_CG, dtype=torch.float32)
    target_tensor = torch.tensor(target_FG, dtype=torch.float32)
    
    return input_tensor, target_tensor
# ----------------------------
# UAFNO
# ----------------------------
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
        self.num_heads = num_heads
        self.patch_size = patch_size

        # Positional Embedding
        self.positional_embedding = nn.Parameter(torch.zeros(1, dim, 1, 1))  # Broadcastable to (B, C, H, W)
        
       # Patch embedding
        self.patch_embed = nn.Conv2d(
            dim, dim, kernel_size=patch_size, stride=patch_size, padding=0, bias=False
        ) if patch_size > 1 else nn.Identity()

        self.norm1 = nn.LayerNorm(2 * dim)
        self.attn = nn.MultiheadAttention(2 * dim, num_heads, batch_first=True)

        # Fully Connected Layers after IFFT
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

        self.act = nn.GELU()

    def forward(self, x):
        # Patch and Positional Embedding
        x_emb = self.patch_embed(x) + self.positional_embedding

        # Store residual connection with embeddings
        residual = x_emb

        # FFT processing
        B, C, H, W = x_emb.shape
        x_fft = torch.fft.rfft2(x_emb, norm='ortho')
        x_combined = torch.cat([x_fft.real, x_fft.imag], dim=1)

        x_tokens = x_combined.permute(0, 2, 3, 1).view(B, -1, 2 * self.dim)
        x_tokens = self.norm1(x_tokens)
        attn_out, _ = self.attn(x_tokens, x_tokens, x_tokens)
        attn_out = attn_out.view(B, H, W // 2 + 1, 2 * self.dim).permute(0, 3, 1, 2)

        # IFFT processing
        x_real, x_imag = torch.chunk(attn_out, 2, dim=1)
        x_ifft = torch.fft.irfft2(torch.complex(x_real, x_imag), s=(H, W), norm='ortho')

        # Fully Connected Layers
        x_fc = self.act(self.fc1(x_ifft.permute(0, 2, 3, 1)))
        x_fc = self.fc2(x_fc).permute(0, 3, 1, 2)

        # Add residual connection
        x_out = x_fc + residual

        return x_out


class SuperResUAFNO(nn.Module):
    def __init__(self):
        super().__init__()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
        self.encoder = Encoder()
        self.bottleneck = nn.Sequential(*[AFNOBlock() for _ in range(12)])
        self.decoder = Decoder()

    def forward(self, x):
        x4, x3, x2, x1 = self.encoder(x)
        x = self.bottleneck(x4)
        return self.decoder(x, (x3, x2, x1))

# ----------------------------
# Training Setup
# ----------------------------
def train_model():
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    best_val_loss = float('inf')
    # Load and prepare data
    print('Loading data')
    input_tensor, target_tensor = load_data()
    print('Data loaded')
    print(f'Shape of inputs: {input_tensor.shape}')
    print(f'Shape of targets: {target_tensor.shape}')
    dataset = TensorDataset(input_tensor, target_tensor)
    train_ds, val_ds = random_split(dataset, [80, 20])
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    
    # Initialize model
    model = SuperResUAFNO().to(device)
    print('Compiling model')
    #model = torch.compile(model)
    print('Model compiled')
    num_epochs = 30000
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.L1Loss()
    
    # Training loop
    for epoch in trange(num_epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
        
        # Validation
        if epoch % 100 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            
            avg_val_loss = val_loss/len(val_loader)
            print(f"Epoch {epoch} | Train Loss: {loss.item():.8f} | Val Loss: {avg_val_loss:.8f}")
        if epoch % 1000 == 0:
            # Save periodic checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': loss.item(),
                'val_loss': avg_val_loss,
            }, f"/pscratch/sd/h/hbassi/GS_UAFNO_checkpoint_epoch_{epoch}.pth")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "/pscratch/sd/h/hbassi/GS_UAFNO_best_model.pth")
                print(f"New best model saved with val loss {avg_val_loss:.8f}")

# ----------------------------
# Run Training
# ----------------------------
if __name__ == "__main__":
    train_model()