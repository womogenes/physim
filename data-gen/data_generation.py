from tqdm import tqdm
import torch
import datetime as dt
import os

OUTPUT_DIR = os.path.abspath("./data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using torch device: {device}")

WIDTH, HEIGHT = 128, 128

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

    epsilon = 20  # For buffering/smoothing effect

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

    v_new = v + a * 0.5
    x_new = x + v_new * dt
    a_new = get_grav_acc(x_new, m, G)
    v_new = v + 0.5 * a_new

    return x_new, v_new


def generate_timeline(x0, v0, m, G, dt, F):
    """
    Generate one timeline of frames for gravity simulation.

    arguments:
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
    # V[0] = v0
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


def points_to_histograms(X, weight, batch_size=64):
    """
    Vectorized histogram creation with batched frames to manage memory.
        i.e. "fast batched histograms"
    
    arguments:
        X:       (F, n, 2) tensor of positions across F frames
        weight:  (F, n) tensor of weights
        batch_size: number of frames to process at once
    """
    F, n, _ = X.shape
    assert tuple(weight.shape) == (F, n), "Weights must be of shape (F, n)"
    assert F % batch_size == 0
    
    result = torch.zeros(F, WIDTH * HEIGHT, device=X.device)
    for i in range(0, F, batch_size):
        batch_end = min(i + batch_size, F)

        X_slice = X[i:batch_end]   # shape: (f, n, 2) where f = batch_end - i
        mask = (0 <= X_slice[:,:,0]) & (X_slice[:,:,0] < WIDTH) \
             & (0 <= X_slice[:,:,1]) & (X_slice[:,:,1] < HEIGHT)
    
        net_weight = torch.zeros(batch_size, HEIGHT * WIDTH)
        
        # Assign flattened indices
        indices = (X_slice[:,:,0].long() * WIDTH + X_slice[:,:,1].long())  # shape: (f, n)
        indices = torch.clamp(indices, 0, WIDTH * HEIGHT - 1)

        net_weight.scatter_add_(1, indices, weight[i:batch_end] * mask)

        # Add back to results, cutting out extra index
        result[i:batch_end] += net_weight

    return result.reshape((F, HEIGHT, WIDTH))


def voxelize_timeline(timeline):
    """
    Converts point-cloud timeline to voxelized timeline.
    Output is (F, width, height, 4) tensor. Each (F, width, height)
        slice contains a 4-vector of features:
            [0] net x-momentum
            [1] net y-momentum
            [2] net mass
            [3] number of particles

    arguments:
        timeline: {
            "dt": dt,
            "G": "G",
            "m": (n,) array
            "X": (F, n, 2) array
            "V": (F, n, 2) array
        }
    """
    dt, G, m, X, V = [timeline[key] for key in ["dt", "G", "m", "X", "V"]]
    F, n, _ = X.shape

    p_x = V[:,:,0] * m[None,:]                  # x-momentum, (F, n)
    p_y = V[:,:,1] * m[None,:]                  # y-momentum, (F, n)
    m = m                                       # masses, (n,)
    ones = torch.ones((F, n))                   # to count objects, (F, n)

    p_x_channel = points_to_histograms(X, weight=p_x)
    p_y_channel = points_to_histograms(X, weight=p_y)
    m_channel = points_to_histograms(X, weight=torch.broadcast_to(m[None,:], (F, n)))
    count_channel = points_to_histograms(X, weight=ones)

    result = torch.stack((p_x_channel, p_y_channel, m_channel, count_channel), dim=1)
    return {
        "dt": dt,
        "G": G,
        "m": m,
        "frames": result,
    }


if __name__ == "__main__":
    F = 512             # Frames per timeline
    dt = 0.1            # Timestep per frame
    G = 20              # Gravitational constant
    n_samples = 10      # Number of timelines to generate

    n = 512             # Number of particles

    data_dir = f"n_{n}_G_{G}_dt_{dt}_F_{F}_leapfrog"
    os.makedirs(f"./data/{data_dir}/cloud", exist_ok=True)
    os.makedirs(f"./data/{data_dir}/voxel", exist_ok=True)

    for i in tqdm(range(n_samples), ncols=80):
        # if os.path.exists(f"{OUTPUT_DIR}/{data_dir}/cloud/{i:>06}.pt"):
        #     continue

        # Generate random positions
        x0 = torch.hstack([torch.rand((n, 1)) * WIDTH, torch.rand((n, 1)) * HEIGHT])

        # Generate random velocities
        v0 = torch.randn((n, 2)) * 2

        # Generate random masses according to log scale
        m = torch.exp(torch.randn((n,)) * 0.5 + 1)

        # Generate timeline
        cloud_timeline = generate_timeline(x0, v0, m, G, dt, F)
        torch.save(cloud_timeline, f"{OUTPUT_DIR}/{data_dir}/cloud/{i:>06}.pt")

        # Generate voxelized timeline
        # voxel_timeline = voxelize_timeline(cloud_timeline)
        # torch.save(voxel_timeline, f"{OUTPUT_DIR}/{data_dir}/voxel/{i:>06}.pt")
