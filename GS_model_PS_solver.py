import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import warnings
from tqdm import trange
# Pseudospectral solver parameters
Nx_fine = 128  # Fine-scale resolution
Ny_fine = 128
Nx_coarse = 48  # Coarse-scale resolution
Ny_coarse = 48
Lx, Ly = 100.0, 100.0  # Domain size
dt = 0.01  # Time step
Nt = 50  # Number of time steps

# Diffusion and reaction parameters
D_u = 0.16
D_v = 0.08
F = 0.04
k = 0.06

# Initialize spatial grids for fine and coarse resolutions
x_fine = np.linspace(0, Lx, Nx_fine, endpoint=False)
y_fine = np.linspace(0, Ly, Ny_fine, endpoint=False)
x_coarse = np.linspace(0, Lx, Nx_coarse, endpoint=False)
y_coarse = np.linspace(0, Ly, Ny_coarse, endpoint=False)

# Generate meshgrid for spatial domain
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
kx_fine = 2 * np.pi * np.fft.fftfreq(Nx_fine, d=(Lx / Nx_fine))
ky_fine = 2 * np.pi * np.fft.fftfreq(Ny_fine, d=(Ly / Ny_fine))
kx_coarse = 2 * np.pi * np.fft.fftfreq(Nx_coarse, d=(Lx / Nx_coarse))
ky_coarse = 2 * np.pi * np.fft.fftfreq(Ny_coarse, d=(Ly / Ny_coarse))

Kx_fine, Ky_fine = np.meshgrid(kx_fine, ky_fine, indexing='ij')
Kx_coarse, Ky_coarse = np.meshgrid(kx_coarse, ky_coarse, indexing='ij')
total_fine_scale_dynamics = []
total_coarse_scale_dynamics = []

# Function to filter overflow warnings
def overflow_warning_filter(record):
    return issubclass(record.category, RuntimeWarning) and "overflow encountered" in str(record.message)

for _ in range(3):
    # Parameters for the random smooth function
    num_terms = 20  # Number of random Fourier terms
    sigma = 5  # Standard deviation for the Gaussian envelope

    # Generate random Fourier coefficients for fine grid
    A_fine = np.random.uniform(-1, 1, (num_terms, num_terms))
    num_elements = num_terms ** 2  # Total elements in the matrix
    num_to_zero = num_elements // 40# Number of elements to zero per iteration
    
    for idx in trange(num_elements + 1):
        if idx == num_elements:
            A_modified = A_fine
        else:
            A_modified = A_fine.copy()
            zero_indices = np.random.choice(num_elements, num_to_zero, replace=False)
            for index in zero_indices:
                i, j = divmod(index, num_terms)
                A_modified[i, j] = 0

        # Create a Gaussian envelope in the spatial domain
        gaussian_envelope_fine = np.exp(-((X_fine - Lx / 2)**2 + (Y_fine - Ly / 2)**2) / (2 * sigma**2))
    
        # Generate random smooth function for U_fine using trigonometric terms and Gaussian envelope
        U_fine = np.zeros_like(X_fine)
    
        for i in range(num_terms):
            for j in range(num_terms):
                U_fine += A_modified[i, j] * np.sin((i + 1) * np.pi * X_fine / Lx) * np.cos((j + 1) * np.pi * Y_fine / Ly)
    
        U_fine *= gaussian_envelope_fine
    
        # Initialize V_fine as zero, or use another function
        V_fine = 0.2 * U_fine
    
        # Generate random Fourier coefficients for the coarse grid
        A_coarse = A_modified
    
        # Create a Gaussian envelope in the spatial domain for coarse grid
        gaussian_envelope_coarse = np.exp(-((X_coarse - Lx / 2)**2 + (Y_coarse - Ly / 2)**2) / (2 * sigma**2))
    
        # Generate random smooth function for U_coarse using trigonometric terms and Gaussian envelope
        U_coarse = np.zeros_like(X_coarse)
    
        for i in range(num_terms):
            for j in range(num_terms):
                U_coarse += A_coarse[i, j] * np.sin((i + 1) * np.pi * X_coarse / Lx) * np.cos((j + 1) * np.pi * Y_coarse / Ly)
    
        U_coarse *= gaussian_envelope_coarse
    
        # Initialize V_coarse as zero, or use another function
        V_coarse = 0.2 * U_coarse
    
        # Store dynamics
        fine_scale_dynamics = np.zeros((Nt, Nx_fine, Ny_fine))
        coarse_scale_dynamics = np.zeros((Nt, Nx_coarse, Ny_coarse))
    
        overflow_detected = False  # Flag to track overflow
    
        # Time-stepping loop
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Capture all warnings
            for n in range(Nt):
                # Fine-scale solver
                U_hat_fine = fft2(U_fine)
                V_hat_fine = fft2(V_fine)
    
                laplacian_fine = -(Kx_fine**2 + Ky_fine**2)
                U_hat_fine += dt * (D_u * laplacian_fine * U_hat_fine - fft2(U_fine * V_fine**2) + fft2(F * (1 - U_fine)))
                V_hat_fine += dt * (D_v * laplacian_fine * V_hat_fine + fft2(U_fine * V_fine**2) - fft2((F + k) * V_fine))
    
                U_fine = np.real(ifft2(U_hat_fine))
                V_fine = np.real(ifft2(V_hat_fine))
                fine_scale_dynamics[n] = U_fine
    
                # Coarse-scale solver
                U_hat_coarse = fft2(U_coarse)
                V_hat_coarse = fft2(V_coarse)
    
                laplacian_coarse = -(Kx_coarse**2 + Ky_coarse**2)
                U_hat_coarse += dt * (D_u * laplacian_coarse * U_hat_coarse - fft2(U_coarse * V_coarse**2) + fft2(F * (1 - U_coarse)))
                V_hat_coarse += dt * (D_v * laplacian_coarse * V_hat_coarse + fft2(U_coarse * V_coarse**2) - fft2((F + k) * V_coarse))
    
                U_coarse = np.real(ifft2(U_hat_coarse))
                V_coarse = np.real(ifft2(V_hat_coarse))
                coarse_scale_dynamics[n] = U_coarse
    
                # Check for overflow warnings
                if any(overflow_warning_filter(record) for record in w):
                    print(f"Overflow detected at time step {n}. Skipping trajectory.")
                    overflow_detected = True
                    break  # Exit the time-stepping loop
    
        if overflow_detected:
            continue  # Skip to the next initial condition
    
        # Normalize both datasets if no overflow occurred
        fine_scale_dynamics -= np.min(fine_scale_dynamics)
        fine_scale_dynamics /= np.max(fine_scale_dynamics)
    
        coarse_scale_dynamics -= np.min(coarse_scale_dynamics)
        coarse_scale_dynamics /= np.max(coarse_scale_dynamics)
    
        fine_scale_dynamics = fine_scale_dynamics[::10]
        coarse_scale_dynamics = coarse_scale_dynamics[::10]
        # plt.imshow(fine_scale_dynamics[0])
        # plt.show()
        # plt.imshow(coarse_scale_dynamics[0])
        # plt.show()
        # plt.imshow(fine_scale_dynamics[-1])
        # plt.show()
        # plt.imshow(coarse_scale_dynamics[-1])
        # plt.show()
        # if total_coarse_scale_dynamics:
        #     print(np.linalg.norm(coarse_scale_dynamics - total_coarse_scale_dynamics[-1]))
        total_fine_scale_dynamics.append(fine_scale_dynamics)
        total_coarse_scale_dynamics.append(coarse_scale_dynamics)
total_fine_scale_dynamics_np = np.array(total_fine_scale_dynamics)
total_coarse_scale_dynamics_np = np.array(total_coarse_scale_dynamics)
with open(f'GS_model_multi_traj_data_fine_scale_dynamics_1200_masked_Aij_tenth_tmax=5_sigma={sigma}_numterms={num_terms}.npy', 'wb') as f:
    np.save(f, total_fine_scale_dynamics_np)
with open(f'GS_model_multi_traj_data_coarse_scale_dynamics_1200_masked_Aij_tenth_tmax=5_sigma={sigma}_numterms={num_terms}.npy', 'wb') as f:
    np.save(f, total_coarse_scale_dynamics_np)