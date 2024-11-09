# N-body simulation data

All data here was generated with a naive O(n^2) n-body solver.

## Generation

Run `./data_generation.py` to create the `./data/` directory. It will generate a number of files:

```
data-gen/data/
  n_512_dt_0.1_F_512/
    cloud/
      - 000000.pt
      - 000001.pt
      - 000002.pt
      ...
    voxel/
      - 000000.pt
      - 000001.pt
      - 000002.pt
      ...
```

Each file describes a timeline (i.e. a single run of $n$ particles). Components of the parent directory's name:

- `n_512` means 512 particles
- `dt_0.1` means the timestep is 0.1 per frame
- `F_512` means 512 frames per timeline

## Cloud files

So-named because they describe the points in a point-cloud format. To load a file using PyTorch:

```py
import torch
data = torch.load("path/to/data/dir/n_512_dt_0.1_F_512_/cloud/000000.pt")
```

`data` will be a dictionary of floats/tensors that describe the timeline. `n` is the number of particles, `dt` is the timestep, `F` is the length of the timeline in frames, and `G` is the gravitational constant:

```py
{
    "dt": 0.1,
    "G": 100,
    "m": tensor(...)  # tensor of masses        shape: (n,)
    "X": tensor(...)  # tensor of positions     shape: (F, n, 2)
    "V": tensor(...)  # tensor of velocities    shape: (F, n, 2)
}
```

## Voxel files

So-named because they describe the points as in image. To load a file using PyTorch:

```py
import torch
data = torch.load("path/to/data/dir/n_512_dt_0.1_F_512_/voxel/000000.pt")
```

`data` will be a dictionary of floats/tensors that describe the timeline. `n` is the number of particles, `dt` is the timestep, `F` is the length of the timeline in frames, and `G` is the gravitational constant:

```py
{
    "dt": 0.1,
    "G": 100,
    "m": tensor(...)        # tensor of masses   shape: (n,)
    "frames": tensor(...)   # tensor of images   shape: (F, WIDTH, HEIGHT, 4)
}
```

The formamt of `data["frames"]` is as a sequence of `F` images, where each image is of dimension `WIDTH x HEIGHT` and has four channels. The channels are:

```
[0] net x-momentum in voxel
[1] net y-momentum in voxel
[2] net mass in cell
[3] number of particles in cell
```
