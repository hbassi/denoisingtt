import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import trange
import random
from torch import save
import os

torch.manual_seed(seed=999)
torch.set_float32_matmul_precision('high')

# Directly use the full dataset
# input_CG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_coarse_scale_dynamics.npy')
# target_FG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_fine_scale_dynamics.npy')
input_CG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_coarse_scale_dynamics_TT.npy')
target_FG = np.load('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_fine_scale_dynamics_TT.npy')

# # Reshape the data to include trajectory and temporal structure
# num_trajectories = 87
# time_steps = 400
# input_CG = input_CG.reshape((num_trajectories, time_steps, 1, 48, 48))
# target_FG = target_FG.reshape((num_trajectories, time_steps, 1, 128, 128))

print(input_CG.shape, target_FG.shape)
input_CG = np.expand_dims(input_CG, 2)
target_FG = np.expand_dims(target_FG, 2)
print(input_CG.shape, target_FG.shape)
# Dataset class
class TrajectoryDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return 1 #self.input_data.shape[0]  # Number of trajectories

    def __getitem__(self, idx):
        # Return the input and target for the specific trajectory
        input_sample = torch.tensor(self.input_data[idx], dtype=torch.float32)  # Shape: [time_steps, 1, 48, 48]
        target_sample = torch.tensor(self.target_data[idx], dtype=torch.float32)  # Shape: [time_steps, 1, 128, 128]
        return input_sample, target_sample

# Create the full dataset
full_dataset = TrajectoryDataset(input_CG, target_FG)

# # Random split into training and validation sets (80-20 split)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
# train_dataset = full_dataset
# val_dataset = full_dataset

# Create DataLoader for both training and validation datasets
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)  # No shuffle as we pick random trajectory
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )

#         self.middle = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU()
#         )

       
#         self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU()
#         )

#         self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU()
#         )

     
#         self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
#         self.dec3 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU()
#         )

#         self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
#         self.dec4 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.ReLU()
#         )

#         # Final decoder layer to bring it back to 1 output channel
#         self.up5 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
#         self.dec5 = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=3, padding=1)
#         )

#     def forward(self, x):
#         # Encoder path
#         enc = self.encoder(x)

#         # Middle path (bottleneck)
#         mid = self.middle(enc)

#         # Decoder path
#         up1 = self.up1(mid)
#         dec1 = self.dec1(up1)
#         up2 = self.up2(dec1)
#         dec2 = self.dec2(up2)
#         up3 = self.up3(dec2)
#         dec3 = self.dec3(up3)
#         up4 = self.up4(dec3)
#         dec4 = self.dec4(up4)
#         up5 = self.up5(dec4)
#         dec5 = self.dec5(up5)

#         # Return the output, truncating to 128x128
#         return dec5[:, :, :, :128]
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
        return dec3[:, :, :100, :100]  # Ensure the final output is 128x128
unet = UNet()

# Training setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device: ', device)
unet = UNet().to(device)
print('Compiling model')
unet = torch.compile(unet)
print('Model compiled')
num_epochs = 30000
criterion = nn.MSELoss()
optimizer = optim.Adam(unet.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# Logging information
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'training_log_large_Unet.txt')

# Open the log file for writing
with open(log_file_path, 'w') as log_file:
    log_file.write("Epoch,Training Loss,Validation Loss\n")  


    for epoch in trange(num_epochs):
        unet.train()
        running_loss = 0.0

        # Select a random trajectory during each epoch for training
        trajectory_idx = np.random.randint(0, 99)
        input_trajectory = input_CG[trajectory_idx]
        target_trajectory = target_FG[trajectory_idx]

        # Create a new DataLoader for this trajectory
        trajectory_dataset = TensorDataset(torch.tensor(input_trajectory, dtype=torch.float32),
                                           torch.tensor(target_trajectory, dtype=torch.float32))
        trajectory_dataloader = DataLoader(trajectory_dataset, batch_size=batch_size, shuffle=False)

        for input_batch, target_batch in trajectory_dataloader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            outputs = unet(input_batch)
            loss = criterion(outputs, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # Validation every 100 epochs
        val_loss = 0.0
        if epoch % 100 == 0:
            unet.eval()
            with torch.no_grad():
                # Select a random trajectory during each validation step
                val_trajectory_idx = random.randint(0, len(val_dataset) - 1)
                val_input_trajectory = input_CG[val_trajectory_idx]
                val_target_trajectory = target_FG[val_trajectory_idx]

                # Create a new DataLoader for this validation trajectory
                val_trajectory_dataset = TensorDataset(torch.tensor(val_input_trajectory, dtype=torch.float32),
                                                        torch.tensor(val_target_trajectory, dtype=torch.float32))
                val_trajectory_dataloader = DataLoader(val_trajectory_dataset, batch_size=batch_size, shuffle=False)

                for val_input_batch, val_target_batch in val_trajectory_dataloader:
                    val_input_batch = val_input_batch.to(device)
                    val_target_batch = val_target_batch.to(device)

                    val_outputs = unet(val_input_batch)
                    val_loss_item = criterion(val_outputs, val_target_batch)
                    val_loss += val_loss_item.item()

            # Write the losses to the log file
            log_file.write(f"{epoch+1},{running_loss / len(trajectory_dataloader):.8f},{val_loss / len(val_trajectory_dataloader):.8f}\n")
            log_file.flush()  # Ensure the log is written immediately

            # Print training and validation loss every 100 epochs
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(trajectory_dataloader):.8f}, Validation Loss: {val_loss / len(val_trajectory_dataloader):.8f}")
            #print(outputs)

        if epoch % 1000 == 0:
            # Save model checkpoint periodically
            checkpoint_path = f'/pscratch/sd/h/hbassi/unet_epoch_{epoch}_GS_multi_traj_largeUnet.pth'
            save(unet.state_dict(), checkpoint_path)
            print(f'Model saved to {checkpoint_path}')
