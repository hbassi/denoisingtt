import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import trange
torch.set_float32_matmul_precision('high')
torch.manual_seed(seed=999)
# ----------------------------
# Data Loading and Preparation
# ----------------------------
def load_data():
    # Load and preprocess data (modify paths as needed)
    input_CG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_coarse_scale_dynamics_1500_tmax=5_sigma=7_numterms=20.npy').reshape((1447, 50, 1, 48, 48))[:1000, :10, 0, :, :]
    target_FG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_fine_scale_dynamics_1500_tmax=5_sigma=7_numterms=20.npy').reshape((1447, 50, 1, 128, 128))[:1000, :10, 0, :, :]
    # Convert to PyTorch tensors
    input_tensor = torch.tensor(input_CG, dtype=torch.float32)
    target_tensor = torch.tensor(target_FG, dtype=torch.float32)
    return input_tensor, target_tensor

# ----------------------------
# Fourier Neural Operator Layer
# ----------------------------
class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(FourierLayer, self).__init__()
        assert in_channels == out_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to use in the first spatial dimension
        self.modes2 = modes2  # Number of Fourier modes to use in the second spatial dimension

        # Scaling factor for weight initialization
        self.scale = (1 / (in_channels * out_channels))
        
        # Learnable complex weights for Fourier transforms
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        residual = x  # Save for residual connection
        
        # Compute 2D Real FFT (output has complex numbers)
        x_ft = torch.fft.rfft2(x)
        batch_size, _, h, w = x_ft.shape

        # Initialize output Fourier tensor
        out_ft = torch.zeros(batch_size, self.out_channels, h, w//2 + 1, device=x.device, dtype=torch.cfloat)
        

        available_modes1 = min(self.modes1, h)
        available_modes2 = min(self.modes2, w//2 + 1)
        
        out_ft[:, :, :available_modes1, :available_modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :available_modes1, :available_modes2], 
            self.weights1[:, :, :available_modes1, :available_modes2]
        )
        out_ft[:, :, -available_modes1:, :available_modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -available_modes1:, :available_modes2],
            self.weights2[:, :, :available_modes1, :available_modes2]
        )

        # Inverse FFT to spatial domain
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        # Add residual connection
        return x + residual
# ----------------------------
# U-Net Architecture
# ----------------------------
class SuperResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder (48x48 -> 24x24)
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        
        # Bottleneck with stabilized Fourier layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            FourierLayer(128, 128, modes1=16, modes2=12),  # Adjusted for 24x24 input
            nn.BatchNorm2d(128),  # Added for stability
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Decoder (24x24 -> 128x128)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 24->48
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 48->96
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(size=128, mode='bilinear'),  # 96->128
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 10, 3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

# ----------------------------
# Training Setup
# ----------------------------
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_val_loss = float('inf')
    # Load and prepare data
    print('Loading data')
    input_tensor, target_tensor = load_data()
    print('Data loaded')
    print(f'Shape of inputs: {input_tensor.shape}')
    print(f'Shape of targets: {target_tensor.shape}')
    dataset = TensorDataset(input_tensor, target_tensor)
    train_ds, val_ds = random_split(dataset, [950, 50])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    # Initialize model
    model = SuperResUNet().to(device)
    print('Compiling model')
    #model = torch.compile(model)
    print('Model compiled')
    num_epochs = 20000
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.L1Loss()
    training_losses = []
    validation_losses = []
    # Training loop
    for epoch in trange(num_epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            training_losses.append(loss.item())
            validation_losses.append(avg_val_loss)
        if epoch % 1000 == 0:
            # Save periodic checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': loss.item(),
                'val_loss': avg_val_loss,
            }, f"/pscratch/sd/h/hbassi/models/GS_unet_checkpoint_epoch_{epoch}_ablation_masked_t=3.pth")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "/pscratch/sd/h/hbassi/models/GS_unet_best_model_ablation_mem=10.pth")
                print(f"New best model saved with val loss {avg_val_loss:.8f}")
    with open('./logs/GS_Funet_multi_traj_training_ablation_masked_mem=10.npy', 'wb') as f:
        np.save(f, training_losses)
    f.close()
    with open('./logs/GS_Funet_multi_traj_validation_ablation_masked_mem=10.npy', 'wb') as f:
        np.save(f, validation_losses)
    f.close()

# ----------------------------
# Run Training
# ----------------------------
if __name__ == "__main__":
    train_model()