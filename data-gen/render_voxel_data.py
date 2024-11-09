# Testing script for rendering sequence of frames represented as images (not point clouds)

import  matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
import cv2
import imageio
from pathlib import Path
from torchvision.transforms.functional import gaussian_blur


file = "n_512_dt_0.1_F_512/voxel/000000.pt"

timeline = torch.load(f"./data/{file}", weights_only=True)
dt, G, m, frames = [timeline[key] for key in ["dt", "G", "m", "frames"]]
F, WIDTH, HEIGHT, _ = frames.shape

sqrt_m = torch.sqrt(m)

frames = gaussian_blur(frames[:,:,:,2], kernel_size=11)

# F is number of frames
# n is number of particles

rendered_frames = []
fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
fig.patch.set_facecolor('black')

for i in tqdm(range(F), ncols=80):
    frame = frames[i]

    ax.clear()
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.imshow(frame, cmap="Blues", vmin=0, vmax=1)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height() + (4,))
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Make the image writable by creating a copy
    image = image.copy()
    rendered_frames.append(image)

plt.close()

print(f"Finished rendering, saving to MP4...")
imageio.mimsave(f"./{Path(file).stem}_net_mass.mp4", rendered_frames, fps=30)
