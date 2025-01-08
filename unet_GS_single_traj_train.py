import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
from tqdm import trange
import random
from torch import save

torch.manual_seed(seed=999)
torch.set_float32_matmul_precision('high')

# Directly use only the first trajectory in the dataset
input_CG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_coarse_scale_dynamics.npy')
target_FG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_fine_scale_dynamics.npy')

# Reshape the data to include trajectory and temporal structure
num_trajectories = 87
time_steps = 400
input_CG = input_CG.reshape((num_trajectories, time_steps, 1, 48, 48))
target_FG = target_FG.reshape((num_trajectories, time_steps, 1, 128, 128))

print(input_CG.shape, target_FG.shape)

# Modify the dataset class to handle reshaping correctly
class SingleTrajectoryDataset(Dataset):
    def __init__(self, input_data, target_data, trajectory_idx=None):
        """
        Initialize with the full dataset and optionally specify which trajectory to use.
        """
        self.input_data = input_data
        self.target_data = target_data
        self.trajectory_idx = trajectory_idx

        # If trajectory_idx is provided, only use that specific trajectory
        if self.trajectory_idx is not None:
            self.input_data = self.input_data[self.trajectory_idx]
            self.target_data = self.target_data[self.trajectory_idx]

    def __len__(self):
        return self.input_data.shape[0]  # Number of time steps

    def __getitem__(self, idx):
        # Return the input and target for the specific time step idx
        input_sample = torch.tensor(self.input_data[idx], dtype=torch.float32)  
        target_sample = torch.tensor(self.target_data[idx], dtype=torch.float32)  
        
        return input_sample, target_sample



# Define the U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
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
        return dec3[:, :, :128, :128]

# Training setup
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using device: ', device)
unet = UNet().to(device)
print('Compiling model')
unet = torch.compile(unet)
print('Model compiled')
criterion = nn.MSELoss()
num_epochs = 30000
optimizer = optim.Adam(unet.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
trajectory_idx = 0
losses = []
# Training loop
for epoch in trange(1, num_epochs + 2):
    unet.train()
    running_loss = 0.0
    #trajectory_idx = np.random.randint(0, num_trajectories - 1)
    train_dataset = SingleTrajectoryDataset(input_CG, target_FG, trajectory_idx=trajectory_idx)
    batch_size = 16

    # Create DataLoader for the selected trajectory
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (input_batch, target_batch) in enumerate(train_dataloader):
        if not batch_idx:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            outputs = unet(input_batch)
            loss = criterion(outputs, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
    
    if epoch % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_dataloader):.8f}")
        losses.append(running_loss)
        # print(losses)
        # print(outputs)
        checkpoint_path = f'/pscratch/sd/h/hbassi/unet_epoch_{epoch}_GS_multi_traj_training.pth'
        save(unet.state_dict(), checkpoint_path)
        print(f'Model saved to {checkpoint_path}')

    # # Save model checkpoint periodically
    if epoch % 15000 == 0:
        trajectory_idx = np.random.randint(0, num_trajectories - 1)
        print(f'New trajectory selected: {trajectory_idx}')
        