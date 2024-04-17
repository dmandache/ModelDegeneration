import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm

# Define a function to calculate the FID
def calculate_fid(real_images, generated_images, batch_size=64):
    # Define a transform to normalize the images
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # Load the pre-trained Inception v3 model
    inception_model = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
    inception_model.eval()
    if torch.cuda.is_available():
        inception_model = inception_model.cuda()

    # Function to extract feature representations
    def extract_features(images):
        feats = []
        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            batch = images[batch_start:batch_end]
            if torch.cuda.is_available():
                batch = batch.cuda()
            with torch.no_grad():
                feat = inception_model(batch)[0].view(batch.size(0), -1)
            feats.append(feat.cpu().numpy())
        feats = np.concatenate(feats, axis=0)
        return feats

    # Extract features for real and generated images
    real_features = extract_features(real_images)
    generated_features = extract_features(generated_images)

    # Calculate mean and covariance for real and generated features
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_generated = np.cov(generated_features, rowvar=False)

    # Compute the squared Frobenius norm between the means
    mean_diff = mu_real - mu_generated
    mean_diff_squared = np.dot(mean_diff, mean_diff)

    # Compute the trace of the product of covariance matrices
    cov_product = sqrtm(sigma_real.dot(sigma_generated))
    if np.iscomplexobj(cov_product):
        cov_product = cov_product.real
    trace = np.trace(sigma_real + sigma_generated - 2 * cov_product)

    # Compute the Fréchet distance
    fid = mean_diff_squared + trace
    return fid

# Example usage
if __name__ == "__main__":
    # Load MNIST dataset
    mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=10000, shuffle=True)

    # Get a batch of real MNIST images
    real_images, _ = next(iter(mnist_loader))
    real_images = real_images.numpy()

    # Example: generate some fake images (you would replace this with your actual generated images)
    num_generated = real_images.shape[0]
    generated_images = np.random.rand(num_generated, 1, 28, 28)

    # Calculate FID
    fid_score = calculate_fid(real_images, generated_images)
    print("FID score:", fid_score)
