from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio

OUTPUT_DIR = os.path.abspath("./data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using torch device: {device}")

WIDTH = 512

torch.manual_seed(0)


def get_grav_acc(x, m, G):
    """
    Calculate gravitational acceleration of all points.
        x: (n, 2) array of positions
        m: (n,) array of masses
        G: gravitational constant
    """
    n = x.shape[0]
    assert x.shape == (n, 3) and m.shape == (n,)

    epsilon = 2  # For buffering/smoothing effect

    # Calculate pairwise displacement vectors (x_i - x_j)
    dx = x[:, None, :] - x[None, :, :]  # Shape: (n, n, 2)
    d = torch.norm(dx, dim=2)

    mapped_masses = m[:, None].expand(n, n)

    d3 = (d**2 + epsilon**2)**1.5
    F = G * dx * mapped_masses[:, :, None] / d3[:, :, None]
    acc = torch.sum(F, dim=0) / m[:, None]

    return acc


def update_system(x, v, m, dt, G):
    """
    Update points due to gravitational attraction.
        x:  (n, 3) array of positions
        v:  (n, 3) array of velocities
        m:  (n,) array of masses
        dt: time step
        G:  gravitational constant
    """
    n = x.shape[0]
    assert x.shape == (n, 3) and v.shape == (n, 3) and m.shape == (n,)

    # Update positions and velocities using Verlet integration
    # https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
    a = get_grav_acc(x, m, G)

    x_new = x + v * dt + 0.5 * a * dt**2
    a_new = get_grav_acc(x_new, m, G)
    v_new = v + 0.5 * (a + a_new) * dt

    return x_new, v_new


def generate_timeline(x0, v0, m, G, dt, F):
    """
    Generate one timeline of frames for gravity simulation.

    arguments:
        x0: (n, 3) array of initial positions
        v0: (n, 3) array of initial velocities
        m:  (n,) array of masses
        G:  gravitational constant
        dt: timestep per frame
        F:  frames to simulate for
    """
    n = x0.shape[0]
    assert x0.shape == (n, 3) and v0.shape == (n, 3) and m.shape == (n,)

    X = torch.zeros((F, n, 3)).to(device)
    V = torch.zeros((F, n, 3)).to(device)

    X[0] = x0
    V[0] = v0
    m = m.to(device)

    for i in range(1, F):
        X[i], V[i] = update_system(X[i-1], V[i-1], m, dt, G)

    return {
        "dt": dt,
        "G": G,
        "m": m.cpu(),
        "X": X.cpu(),
        "V": V.cpu(),
    }


if __name__ == "__main__":
    F = 512             # Frames per timeline
    dt = 0.1            # Timestep per frame
    G = 200             # Gravitational constant
    n_samples = 10      # Number of timelines to generate

    n = 512             # Number of particles

    data_dir = f"n_{n}_G_{G}_dt_{dt}_F_{F}"
    os.makedirs(f"./data/{data_dir}/cloud", exist_ok=True)

    for i in tqdm(range(n_samples), ncols=80):
        # if os.path.exists(f"{OUTPUT_DIR}/{data_dir}/cloud/{i:>06}.pt"):
        #     continue

        # Generate random positions
        x0 = torch.hstack([torch.rand((n, 3)) * WIDTH])

        # Generate random velocities
        v0 = torch.randn((n, 3)) * 2

        # Generate random masses according to log scale
        m = torch.exp(torch.randn((n,)) * 0.5 + 1)

        # Generate timeline
        cloud_timeline = generate_timeline(x0, v0, m, G, dt, F)
        torch.save(cloud_timeline, f"{OUTPUT_DIR}/{data_dir}/cloud/{i:>06}.pt")
