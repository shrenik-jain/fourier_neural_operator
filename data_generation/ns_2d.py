import torch
import math
import matplotlib.pyplot as plt
import matplotlib
from random_fields import GaussianRF
from timeit import default_timer
import scipy.io


def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    '''
    Parameters:
        w0: initial vorticity
        f: forcing function
        visc: viscosity (1/Re)
        T: final time
        delta_t: internal time-step for solve (decrease if blow-up)
        record_steps: number of in-time snapshots to record
    '''
    # Grid size - must be power of 2
    N = w0.size()[-1]

    # Maximum frequency
    k_max = math.floor(N/2.0)

    # Number of steps to reach final time T
    steps = math.ceil(T/delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.fft2(w0)

    # Forcing to Fourier space
    f_h = torch.fft.fft2(f)

    # If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = f_h.unsqueeze(0)

    # Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat(
        (
            torch.arange(0, k_max, device=w0.device),
            torch.arange(-k_max, 0, device=w0.device)
        ),
        0
    ).repeat(N, 1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)

    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0  # Avoid division by zero
    lap = lap.unsqueeze(0)  # For broadcasting

    # Dealiasing mask
    dealias = (
        (torch.abs(k_y) <= (2.0 / 3.0) * k_max)
        & (torch.abs(k_x) <= (2.0 / 3.0) * k_max)
    ).float()
    dealias = dealias.unsqueeze(0)  # For broadcasting

    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in range(steps):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        # Velocity field in x-direction = psi_y
        q_h = (1j) * 2 * math.pi * k_y * psi_h
        q = torch.fft.ifft2(q_h).real

        # Velocity field in y-direction = -psi_x
        v_h = (-1j) * 2 * math.pi * k_x * psi_h
        v = torch.fft.ifft2(v_h).real

        # Partial x of vorticity
        w_x_h = (1j) * 2 * math.pi * k_x * w_h
        w_x = torch.fft.ifft2(w_x_h).real

        # Partial y of vorticity
        w_y_h = (1j) * 2 * math.pi * k_y * w_h
        w_y = torch.fft.ifft2(w_y_h).real

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F = q * w_x + v * w_y
        F_h = torch.fft.fft2(F)
        # Dealias
        F_h = dealias * F_h

        # Crank-Nicolson update
        denom = 1.0 + 0.5 * delta_t * visc * lap
        numer = (
            -delta_t * F_h
            + delta_t * f_h
            + (1.0 - 0.5 * delta_t * visc * lap) * w_h
        )
        w_h = numer / denom

        # Update real time (used only for recording)
        t += delta_t

        if (j + 1) % record_time == 0:
            # Solution in physical space
            w = torch.fft.ifft2(w_h).real

            # Record solution and time
            sol[..., c] = w
            sol_t[c] = t

            c += 1

    return sol, sol_t


# Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Runnig on Device", device)
print("PyTorch CUDA version:", torch.version.cuda)
print("Number of CUDA devices:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")


# Resolution
s = 64

# Number of solutions to generate
N = 20

# Set up 2d GRF with covariance parameters
GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t_space = torch.linspace(0, 1, s + 1, device=device)[:-1]
X, Y = torch.meshgrid(t_space, t_space, indexing='ij')
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

# Number of snapshots from solution
record_steps = 200

# Inputs
a = torch.zeros(N, s, s, device=device)
# Solutions
u = torch.zeros(N, s, s, record_steps, device=device)

## Solve equations in batches
# Batch size
bsize = 20

c = 0
t0 = default_timer()
for j in range(N // bsize):
    # Sample random fields (initial vorticity)
    w0 = GRF.sample(bsize)

    # Solve NS
    # w0: initial vorticity
    # f: forcing term
    # 1e-3: viscosity
    # 50.0: final time
    # 1e-4: time-step
    # record_steps: number of in-time snapshots to record
    sol, sol_t = navier_stokes_2d(w0, f, 1e-3, 50.0, 1e-4, record_steps)

    a[c:(c + bsize), ...] = w0
    u[c:(c + bsize), ...] = sol

    c += bsize
    t1 = default_timer()
    print(j, c, t1 - t0)

# Save data
print("Saving Files...")
scipy.io.savemat(f'ns_data_R{s}_S{N}.mat', {
    'a': a.cpu().numpy(),
    'u': u.cpu().numpy(),
    't': sol_t.cpu().numpy()
})