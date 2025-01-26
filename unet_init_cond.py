import itertools as it
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset, random_split
import attention as att
import torch.optim as optim
from tqdm import trange
from matplotlib.colors import LogNorm
# For same initialization of weights/biases of model
torch.manual_seed(seed=999)
torch.set_float32_matmul_precision('high')
input_CG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_coarse_scale_dynamics.npy')
target_FG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_fine_scale_dynamics.npy') 

input_CG = input_CG.reshape((87, 400, 1, 48, 48))[:, 0, :, :, :]
target_FG = target_FG.reshape((87, 400, 1, 128, 128))[:, 0, :, :, :]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduces size from 48x48 to 24x24
        )

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Upsample from 12x12 to 24x24
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # Upsample from 24x24 to 48x48
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # Upsample from 48x48 to 96x96
        self.dec3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        enc = self.encoder(x)
        mid = self.middle(enc)
        up1 = self.up1(mid)
        dec1 = self.dec1(up1)
        up2 = self.up2(dec1)
        dec2 = self.dec2(up2)
        up3 = self.up3(dec2)
        dec3 = self.dec3(up3)
        # Crop or pad the output tensor to match the target size (128x128)
        return dec3[:, :, :128, :128]  # Ensure the final output is 128x128
unet = UNet()
device = 'cuda:0'
input_CG_tensor = torch.tensor(input_CG, dtype=torch.float32).to(device)
target_FG_tensor = torch.tensor(target_FG, dtype=torch.float32).to(device)
# Create the full dataset
train_window = 85
#train_window = 700
dataset = TensorDataset(input_CG_tensor[:train_window], target_FG_tensor[:train_window])

# 80% training, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(unet.parameters(), lr=0.001)
# Training loop
num_epochs = 5000
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
print('compiling model')
unet = unet.to(device)
unet = torch.compile(unet)
print('model compiled')
from torch import save
train_losses = []
validation_losses = []
for epoch in trange(0, num_epochs):
    unet.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_dataloader):
    
        output = unet(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    scheduler.step()

    if epoch % 100 == 0:
        # Log training loss
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_dataloader):.8f}')
        #print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_dataloader):.8f}')
        train_losses.append(running_loss / len(train_dataloader))
        print(output)
        print(target)

        # Validation phase
        unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
               
                output = unet(data)
                loss = criterion(output, target)
                val_loss += loss.item()

        print(f'Validation Loss: {val_loss / len(val_dataloader):.8f}')
        validation_losses.append(val_loss / len(val_dataloader))
        #print(f'Validation Loss: {val_loss:.8f}')
        print('=============================================================================')
        
# # Save model checkpoint
# checkpoint_path = f'unet_epoch_{epoch}.pth'
# save(unet.state_dict(), checkpoint_path)
# print(f'Model saved to {checkpoint_path}')
