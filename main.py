import torch
from torch.utils.data import Dataset
from torchvision import datasets
from pythae.data.datasets import DatasetOutput
from pythae.models import RHVAEConfig, RHVAE
from pythae.trainers import BaseTrainerConfig, BaseTrainer
from pythae.samplers import NormalSampler, GaussianMixtureSampler, GaussianMixtureSamplerConfig
from pythae.samplers import RHVAESampler, RHVAESamplerConfig
import wandb
import argparse
import random


def sample_indices(vector, k, seed=None):
    # # Example usage:
    # vector = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3])  # Example tensor with 4 labels
    # k = 3  # Number of points to sample from each label
    # seed = 42  # Seed for reproducibility
    # sampled_indices = sample_indices(vector, k, seed)
    # print("Sampled indices:", sampled_indices)

    if seed is not None:
        random.seed(seed)

    indices = []
    label_dict = {}

    # Group indices by label
    for i, label in enumerate(vector):
        label = label.item() if torch.is_tensor(label) else label
        if label not in label_dict:
            label_dict[label] = [i]
        else:
            label_dict[label].append(i)

    # Sample k points from each label
    for label, label_indices in label_dict.items():
        sampled_indices = random.sample(label_indices, min(k, len(label_indices)))
        indices.extend(sampled_indices)

    random.shuffle(indices)

    return indices


class MNIST(Dataset):
    def __init__(self, data):
        self.data = data.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return DatasetOutput(data=x)
    
if __name__ == 'main':

    # Initialize Weights & Biases
    wandb.init(project='model-degeneration')

    # Argument Parser
    parser = argparse.ArgumentParser(description='Train a RHVAE with synthetic data generation.')
    parser.add_argument('--input_dim', type=int, default=28, help='Dimensionality of the input data')
    parser.add_argument('--latent_dim', type=int, default=8, help='Dimensionality of the latent space')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of degenerating runs')
    parser.add_argument('--n_train', type=int, default=20, help='Number of training samples per class')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test samples per class')
    parser.add_argument('--k', type=int, default=200, help='Number of synthetic data samples to generate at each iteration')
    parser.add_argument('--sampler', choices=['normal', 'gmm', 'rhvae'], default='uniform', help='Sampler type for generating synthetic data')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs for each run')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Real data loader
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    
    train_indeces = sample_indices(mnist_trainset.targets, k=args.n_train, seed=42)
    remaining_indeces = list(set(range(len(mnist_trainset.targets)))-set(train_indeces))
    eval_indeces = sample_indices(mnist_trainset.targets[remaining_indeces], k=args.n_test, seed=42)

    train_dataset = MNIST(mnist_trainset.data[train_indeces].reshape(-1, 1, 28, 28) / 255.)
    eval_dataset = MNIST(mnist_trainset.data[eval_indeces].reshape(-1, 1, 28, 28) / 255.)
    print(train_dataset.data.shape, eval_dataset.data.shape)

    wandb.log({"Real Training Data": wandb.Image(train_dataset),
               "Real Evaluiation Data": wandb.Image(eval_dataset)})

    # Training loop
    for i in range(args.n_runs):

        # Config VAE
        model_config = RHVAEConfig(
            input_dim=(1, 28, 28),
            latent_dim=8
        )

        model = RHVAE(
            model_config=model_config
        )

        # Train VAE
        training_config = BaseTrainerConfig(
            output_dir='experiments',
            num_epochs=args.n_epochs,
            learning_rate=3e-3,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
        )

        trainer = BaseTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config
        )

        trainer.train()

        # Generate synthetic data
        if args.sampler == 'normal':
            sampler = NormalSampler(
                model=model,
                sampler_config=None
            )
        elif args.sampler == 'gmm':
            gmm_sampler_config = GaussianMixtureSamplerConfig(
                n_components=10
            )

            gmm_sampler = GaussianMixtureSampler(
                sampler_config=gmm_sampler_config,
                model=model
            )

            gmm_sampler.fit(
                train_data=train_dataset.data
            )
        elif args.sampler == 'rhvae':
            rh_sampler_config = RHVAESamplerConfig(
            )

            rh_sampler = RHVAESampler(
                sampler_config=None,
                model=model
            )

            rh_sampler.fit(
                train_data=train_dataset.data
            )

        gen_data = sampler.sample(
            num_samples=args.k,
        )

        # Log synthetic data and training loss to Weights & Biases
        wandb.log({"Synthetic Data": wandb.Image(gen_data, caption=f"Iteration {i+1}")})
