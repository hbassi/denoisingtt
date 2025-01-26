import itertools as it
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
from tqdm import trange
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.ndimage import zoom
# For same initialization of weights/biases of model
torch.manual_seed(seed=999)
torch.set_float32_matmul_precision('high')
print('loading data')
input_CG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_coarse_scale_dynamics_1500_tmax=5_sigma=7_numterms=20.npy').reshape((1447, 50, 1, 48, 48))[:1000, :5, 0, :, :]
target_FG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_fine_scale_dynamics_1500_tmax=5_sigma=7_numterms=20.npy').reshape((1447, 50, 1, 128, 128))[:1000, :5, 0, :, :]
input_CG[:, 2, :, :] = 0.0
#input_CG = np.load('/pscratch/sd/h/hbassi/2d_vlasov_data_coarse_scale_64.npy')
#target_FG = np.load('/pscratch/sd/h/hbassi/2d_vlasov_data_coarse_scale_128.npy') 
print('data loaded')
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
# class SuperResUNet(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # Encoder (48x48 -> 24x24)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.GELU(),
#             nn.MaxPool2d(2)
#         )
        
#         # Bottleneck with stabilized Fourier layer
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.GELU(),
#             FourierLayer(128, 128, modes1=16, modes2=12),  # Adjusted for 24x24 input
#             nn.BatchNorm2d(128),  # Added for stability
#             nn.GELU(),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.ReLU()
#         )
        
#         # Decoder (24x24 -> 128x128)
#         self.decoder = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear'),  # 24->48
#             nn.Conv2d(128, 64, 3, padding=1),
#             nn.GELU(),
#             nn.Upsample(scale_factor=2, mode='bilinear'),  # 48->96
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.GELU(),
#             nn.Upsample(size=128, mode='bilinear'),  # 96->128
#             nn.Conv2d(64, 32, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(32, 1, 3, padding=1)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.bottleneck(x)
#         x = self.decoder(x)
#         return x
# class SuperResUNet(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # Encoder (192x192 -> 96x96)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.GELU(),
#             nn.MaxPool2d(2)  # Reduces spatial dim by half
#         )
        
#         # Bottleneck with adjusted Fourier modes
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.GELU(),
#             FourierLayer(128, 128, modes1=48, modes2=24),  # Key adjustment
#             nn.BatchNorm2d(128),
#             nn.GELU(),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.ReLU()
#         )
        
#         # Decoder (96x96 -> 384x384)
#         self.decoder = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear'),  # 96→192
#             nn.Conv2d(128, 64, 3, padding=1),
#             nn.GELU(),
#             nn.Upsample(scale_factor=2, mode='bilinear'),  # 192→384
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(64, 32, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(32, 1, 3, padding=1)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.bottleneck(x)
#         x = self.decoder(x)
#         return x
class SuperResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder (48x48 -> 24x24)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 64, 3, padding=1),
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
            nn.Conv2d(32, 5, 3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

device = 'cpu'
input_CG_tensor = torch.tensor(input_CG, dtype=torch.float32).to(device)
target_FG_tensor = torch.tensor(target_FG, dtype=torch.float32).to(device)

unet = SuperResUNet().to(device)
print('Compiling model')
#unet = torch.compile(unet)
print('Model compiled')
print('Loading weights')
#import pdb; pdb.set_trace()
unet.load_state_dict(torch.load('/pscratch/sd/h/hbassi/models/GS_unet_checkpoint_epoch_7000_ablation_masked_t=3.pth')['model_state_dict'])
print('Weights loaded')
unet.eval().cpu()
inputs = input_CG[:]
print('Making predictions...')
predictions = unet(torch.Tensor(inputs).float()).detach().numpy()
print('Predictions made!')
targets = target_FG[:]

mae_preds = []
mae_interps = []
indices_used =  []

# Example input data dimensions
num_samples = 4
indices = [298] * 4
#indices =  [1, 15, 27, 63]
print(indices)
plt.figure(figsize=(15, 20))

for i, idx in enumerate(indices):
    # Original data
    input_img = inputs[idx, i]  # Shape (48, 48)
    target_img = targets[idx, i]  # Shape (128, 128)
    pred_img = predictions[idx, i]  # Shape (128, 128)
    
    # # Generate interpolated version using SciPy
    zoom_factor = 128/48
    interpolated = zoom(input_img, (zoom_factor, zoom_factor), order=1)
    interpolated = interpolated[:128, :128]  # Ensure exact size
    interp_error = np.abs(interpolated - target_img)
    
    # Calculate positions for subplots (now 6 rows)
    row_positions = [0, 1, 2, 3, 4, 5]
    
    # Plot the input
    ax = plt.subplot(6, num_samples, i + 1 + row_positions[0]*num_samples)
    im = ax.imshow(input_img, cmap='hot')
    plt.title(f'Input {i}')
    plt.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot the target
    ax = plt.subplot(6, num_samples, i + 1 + row_positions[1]*num_samples)
    im = ax.imshow(target_img, cmap='hot')
    plt.title('Target')
    plt.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot the prediction
    ax = plt.subplot(6, num_samples, i + 1 + row_positions[2]*num_samples)
    im = ax.imshow(pred_img, cmap='hot')
    plt.title('Prediction')
    plt.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Prediction error
    ax = plt.subplot(6, num_samples, i + 1 + row_positions[3]*num_samples)
    pred_error = np.abs(pred_img - target_img)
    im = ax.imshow(pred_error, cmap='viridis', norm=LogNorm(vmin=1e-10, vmax=1e-1))
    plt.title('Pred Error')
    plt.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

   # # Plot interpolated
    ax = plt.subplot(6, num_samples, i + 1 + 4*num_samples)
    im = ax.imshow(interpolated, cmap='hot')
    plt.title('Interpolated')
    plt.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Interpolation error
    ax = plt.subplot(6, num_samples, i + 1 + 5*num_samples)
    im = ax.imshow(interp_error, cmap='viridis', norm=LogNorm(vmin=1e-10, vmax=1e-1))
    plt.title('Interp Error')
    plt.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Calculate MAEs
    mae_pred = np.mean(pred_error)
    mae_interp = np.mean(interp_error)
    print(f"Sample {i+1} | Pred MAE: {mae_pred:.4f} | Interp MAE: {mae_interp:.4f} | Improvement: {(mae_interp-mae_pred)/mae_interp*100:.1f}%")
    #print(f"Sample {i+1} | Pred MAE: {mae_pred:.4f}") 
    # Store MAE values
    mae_preds.append(mae_pred)
    mae_interps.append(mae_interp)
    indices_used.append(str(idx))
plt.tight_layout()
plt.savefig(f'./figures/UAFNO_GS_model_training_result_Tmax=5_sigma=7_numterms=20_multi_traj_dataset_randomized_IC_ablation_t=3_masked.png')
plt.close()
# Create bar plot
plt.figure(figsize=(10, 6))
bar_width = 0.35
x_pos = np.arange(len(indices_used))

# # Plot bars
pred_bars = plt.bar(x_pos - bar_width/2, mae_preds, bar_width, 
                   label='Model MAE', color='royalblue')
interp_bars = plt.bar(x_pos + bar_width/2, mae_interps, bar_width, 
                     label='Interpolation MAE', color='lightcoral')

# # Annotate improvement percentages
# for i, (pred, interp) in enumerate(zip(mae_preds, mae_interps)):
#     improvement = (interp - pred)/interp * 100
#     plt.text(x_pos[i], max(pred, interp) + 0.001, 
#             f'+{improvement:.1f}%', 
#             ha='center', va='bottom', fontsize=9)

# Customize plot
plt.xticks(x_pos, indices_used)
plt.xlabel('Sample Index')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Error Comparison: Model vs Naive Interpolation')
plt.legend()
plt.ylim(0, max(max(mae_preds), max(mae_interps)) * 1.2)
plt.tight_layout()
plt.savefig(f'./figures/UAFNO_GS_model_training_result_Tmax=5_sigma=7_numterms=20_multi_traj_dataset_randomized_IC_bar_plot_ablation_t=3_masked.png')