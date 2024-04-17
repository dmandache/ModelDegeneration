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
sample_indeces = sample_indices(real_images.targets, k=200, seed=42)
real_images = real_images.data[sample_indeces].reshape(-1, 1, input_dim, input_dim,) / 255.

generated_images = np.load('experiments/2024-04-17_13-39-41/gendata_0.npy')
generated_images = torch.from_numpy(generated_images)


fid = FrechetInceptionDistance(feature=64)
# generate two slightly overlapping image intensity distributions
# imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
# imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
fid.update(real_images, real=True)
fid.update(generated_images, real=False)
fid_score = fid.compute()

print(real_images.shape, generated_images.shape)

#fid_score = calculate_fid(real_images, generated_images)

print(fid_score)