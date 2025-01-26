import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load training losses
training_losses = np.load('./logs/GS_Funet_multi_traj_training.npy')

# Plot the training losses
plt.figure()
plt.semilogy(training_losses)

# Customize x-axis labels to be multiplied by 100
def multiply_by_100(x, _):
    return f"{int(x * 100)}"

plt.gca().xaxis.set_major_formatter(FuncFormatter(multiply_by_100))

# Add labels and save the plot
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.savefig('training_losses_modified.png')
plt.show()