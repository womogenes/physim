# Testing script for rendering sequence of frames represented as images (not point clouds)

import  matplotlib.pyplot as plt
from tqdm import tqdm
import torch


file = "dt_0.1_F_512_000000.pt"

timeline = torch.load(f"./data/{file}", weights_only=True)
dt, G, m, X, _ = [timeline[key] for key in ["dt", "G", "m", "X", "V"]]

X = X.cpu()
sqrt_m = torch.sqrt(m)

F, n, _ = X.shape
# F is number of frames
# n is number of particles

x_min = 0
x_max = 512
y_min = 0
y_max = 512

frames = []
fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
fig.patch.set_facecolor('black')

F, _, _ = hist.shape

for i in tqdm(range(F), ncols=80):
    x = X[i]

    ax.clear()
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.imshow(hist[i], cmap="Greys_r")
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height() + (4,))
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Make the image writable by creating a copy
    image = image.copy()
    frames.append(image)

plt.close()

print(f"Finished rendering, saving to MP4...")

# Save frames as an animated GIF with looping
imageio.mimsave(f"./test.mp4", frames, fps=30) #, loop=0)