import numpy as np
import torch
from torchvision import datasets
from utils.data import sample_indices
from utils.metrics import calculate_fid

_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance

input_dim = 28

# real data
real_images = datasets.MNIST(root='./data', train=True, download=True, transform=None)
sample_indeces = sample_indices(real_images.targets, k=10, seed=42)
real_images = real_images.data[sample_indeces].reshape(-1, 1, input_dim, input_dim,) / 255
#real_images = real_images.to(dtype=torch.uint8)


generated_images = np.load('experiments/2024-04-23_10-53-54/gendata_0.npy')
generated_images = torch.from_numpy(generated_images)
#generated_images = (generated_images*255).to(dtype=torch.uint8)

# print("Real images:", real_images.shape, real_images.dtype, real_images.min(), real_images.max(), real_images.mean())
# print("Generated images:", generated_images.shape, generated_images.min(), generated_images.max(), generated_images.mean())


fid = FrechetInceptionDistance(feature=64, reset_real_features=False, normalize=True)
# generate two slightly overlapping image intensity distributions
# imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
# imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
fid.update(real_images.expand(real_images.shape[0], 3, 28, 28), real=True)
fid.update(generated_images.expand(generated_images.shape[0], 3, 28, 28), real=False)
fid_score = fid.compute().item()
print(f"FID = {fid_score:.4f}")

import matplotlib.pyplot as plt
from skimage.util import montage

n_images_plot = 25

# Randomly select 9 images from the stack
selected_real_images = real_images[np.random.choice(real_images.shape[0], n_images_plot, replace=False)].squeeze().numpy()
selected_generated_images = generated_images[np.random.choice(generated_images.shape[0], n_images_plot, replace=False)].squeeze().numpy()

# Create a figure and two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(f"FID score = {fid_score:.4f}", fontsize=16)

axes[0].imshow(montage(selected_real_images, grid_shape=(np.sqrt(n_images_plot), np.sqrt(n_images_plot))), cmap='gray')  # Permute dimensions for matplotlib
axes[0].set_title(f"Real Images")
axes[0].axis('off')

axes[1].imshow(montage(selected_generated_images, grid_shape=(np.sqrt(n_images_plot), np.sqrt(n_images_plot))), cmap='gray')  # Permute dimensions for matplotlib
axes[1].set_title(f"Generated Images")
axes[1].axis('off')

plt.tight_layout()
plt.show()