import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# Parameters
L = 100.0          # Length of the domain
Nx = 100           # Number of grid points in x
Ny = 100           # Number of grid points in y
Nt = 1500          # Number of time steps
D_u = 0.16         # Diffusion coefficient for U
D_v = 0.08         # Diffusion coefficient for V
# Carry the feed rate between 0.029 and 0.045
# F = 0.045        # Feed rate of U
k = 0.06           # Kill rate of V
dt = 1.0           # Time step
dx = L / (Nx - 1)  # Spatial step (same for x and y)

# Low-rank approximation function using SVD
def low_rank_approximation(matrix, rank):
    try:
        U, S, VT = np.linalg.svd(matrix, full_matrices=False)
        S = np.diag(S[:rank])
        return U[:, :rank] @ S @ VT[:rank, :]
    except np.linalg.LinAlgError as e:
        print(f"SVD convergence error: {e}")
        return matrix  # Return the unmodified matrix in case of SVD failure

# Laplacian operator for 2D
def compute_laplacian(Z, dx):
    laplacian = (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
                 np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z) / dx**2
    return laplacian

low_rank_dynamics = []
high_rank_dynamics = []

Fs = np.linspace(0.029, 0.045, 500)
ranks = [8, 32]
for F_val in Fs:
    for rank_val in ranks:
        try:
            # Initialize U and V
            U = np.ones((Nx, Ny))
            V = np.zeros((Nx, Ny))
            start, end = 40, 60
            U[start:end, start:end] = 0.4
            V[start:end, start:end] = 0.25
            # Arrays to store dynamics
            dynamics_U = np.zeros((Nt, Nx, Ny))
            dynamics_V = np.zeros((Nt, Nx, Ny))
            F = F_val
            # Low-rank solver
            rank = rank_val  # Desired rank for approximation
            for n in trange(Nt):
                # Compute laplacians
                laplacian_U = compute_laplacian(U, dx)
                laplacian_V = compute_laplacian(V, dx)

                # Update U and V
                U_new = U + (D_u * laplacian_U - U * V**2 + F * (1 - U)) * dt
                V_new = V + (D_v * laplacian_V + U * V**2 - (F + k) * V) * dt

                # Apply low-rank approximation to U and V
                U = low_rank_approximation(U_new, rank)
                V = low_rank_approximation(V_new, rank)
                # Store dynamics
                dynamics_U[n] = U
                dynamics_V[n] = V
            if rank == 8:
                low_rank_dynamics.append(dynamics_U)
            else:
                high_rank_dynamics.append(dynamics_U)
        except Exception as e:
            print(f"Error in trajectory with F={F_val}, rank={rank_val}: {e}")
            continue

with open('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_coarse_scale_dynamics_TT_bd=8.npy', 'wb') as f:
    np.save(f, np.array(low_rank_dynamics))
with open('/pscratch/sd/h/hbassi/GS_model_multi_traj_data_fine_scale_dynamics_TT_bd=32.npy', 'wb') as f:
    np.save(f, np.array(high_rank_dynamics))
