from tqdm import tqdm
import torch
import datetime as dt
import os


OUTPUT_DIR = os.path.abspath("./data")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using torch device: {device}")

torch.manual_seed(0)

def get_grav_acc(x, m, G):
    """
    Calculate gravitational acceleration of all points.
        x: (n, 2) array of positions
        m: (n,) array of masses
        G: gravitational constant
    """
    n = x.shape[0]
    assert x.shape == (n, 2) and m.shape == (n,)

    epsilon = 5  # For buffering/smoothing effect

    # Calculate pairwise displacement vectors (x_i - x_j)
    dx = x[:, None, :] - x[None, :, :]  # Shape: (n, n, 2)
    d = torch.norm(dx, dim=2)

    mapped_masses = m[:, None].expand(n, n)

    F = G * dx * mapped_masses[:, :, None] / (d**3 + epsilon)[:, :, None]
    acc = torch.sum(F, dim=0) / m[:, None]

    return acc


def update_system(x, v, m, dt, G):
    """
    Update points due to gravitational attraction.
        x:  (n, 2) array of positions
        v:  (n, 2) array of velocities
        m:  (n,) array of masses
        dt: time step
        G:  gravitational constant
    """
    n = x.shape[0]
    assert x.shape == (n, 2) and v.shape == (n, 2) and m.shape == (n,)
    
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
        x0: (n, 2) array of initial positions
        v0: (n, 2) array of initial velocities
        m:  (n,) array of masses
        G:  gravitational constant
        dt: timestep per frame
        F:  frames to simulate for
    """
    n = x0.shape[0]
    assert x0.shape == (n, 2) and v0.shape == (n, 2) and m.shape == (n,)

    X = torch.zeros((F, n, 2)).to(device)
    V = torch.zeros((F, n, 2)).to(device)

    X[0] = x0
    V[0] = v0
    m = m.to(device)

    for i in range(1, F):
        X[i], V[i] = update_system(X[i-1], V[i-1], m, dt, G)
    
    return {
        "dt": dt,
        "G": G,
        "m": m,
        "X": X,
        "V": V
    }


if __name__ == "__main__":
    # Generate a bunch of samples
    F = 512
    dt = 0.1
    G = 20
    n_samples = 10

    n = 512  # Number of particles
    width = 512
    height = 512

    os.makedirs("./data", exist_ok=True)

    for i in tqdm(range(n_samples), ncols=80):
        # Generate random positions
        x0 = torch.hstack([torch.rand((n, 1)) * width, torch.rand((n, 1)) * height])

        # Generate random velocities
        v0 = torch.randn((n, 2)) * 3

        # Generate random masses according to log scale
        m = torch.exp(torch.randn((n,)) + 1)

        timeline = generate_timeline(x0, v0, m, G, dt, F)
        torch.save(timeline, f"{OUTPUT_DIR}/dt_{dt}_F_{F}_{i:>06}.pt")
