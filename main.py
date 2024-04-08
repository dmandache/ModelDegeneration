import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets
from pythae.data.datasets import DatasetOutput
from pythae.models import RHVAEConfig, RHVAE
from pythae.trainers import BaseTrainerConfig, BaseTrainer
from pythae.pipelines.training import TrainingPipeline
from pythae.trainers.training_callbacks import WandbCallback
from pythae.samplers import NormalSampler, GaussianMixtureSampler, GaussianMixtureSamplerConfig
from pythae.samplers import RHVAESampler, RHVAESamplerConfig
import wandb
import argparse
import random
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
    
if __name__ == '__main__':

    # Argument Parser
    parser = argparse.ArgumentParser() #description='Train a RHVAE with synthetic data generation.')
    parser.add_argument('--input_dim', type=int, default=28, help='Dimensionality of the input data')
    parser.add_argument('--latent_dim', type=int, default=8, help='Dimensionality of the latent space')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of degenerating runs')
    parser.add_argument('--n_train', type=int, default=20, help='Number of training samples per class')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test samples per class')
    parser.add_argument('--k', type=int, default=200, help='Number of synthetic data samples to generate at each iteration')
    parser.add_argument('--sampler', choices=['normal', 'gmm', 'rhvae'], default='rhvae', help='Sampler type for generating synthetic data')
    parser.add_argument('--model', choices=['CNN', 'MLP'], default='MLP', help='Model Architecture')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of training epochs for each run')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model Config
    model_config = RHVAEConfig(
        input_dim=(1, args.input_dim, args.input_dim),
        latent_dim=args.latent_dim
    )

    # Training Config
    training_config = BaseTrainerConfig(
        output_dir='experiments',
        num_epochs=args.n_epochs,
        learning_rate=3e-3,
        per_device_train_batch_size=100,
        per_device_eval_batch_size=64,
    )

    # Real data loader
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    
    train_indeces = sample_indices(mnist_trainset.targets, k=args.n_train, seed=42)
    remaining_indeces = list(set(range(len(mnist_trainset.targets)))-set(train_indeces))
    eval_indeces = sample_indices(mnist_trainset.targets[remaining_indeces], k=args.n_test, seed=42)

    train_dataset = mnist_trainset.data[train_indeces].reshape(-1, 1, 28, 28) / 255.
    eval_dataset = mnist_trainset.data[eval_indeces].reshape(-1, 1, 28, 28) / 255.
    print(train_dataset.shape, eval_dataset.shape)


    # Training loop
    for i in range(args.n_runs):

        print(f"RUN {i}")

        # W&B init
        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config=training_config, # training config
            model_config=model_config, # model config
            project_name="model-degeneration", # specify your wandb project
        )

        wandb.run.name = f"experiment_{timestamp}_run_{i}"
        wandb.config.update(args)

        callbacks = []
        callbacks.append(wandb_cb)

        wandb.log({"Training Data": wandb.Image(train_dataset),
                   "Evaluation Data": wandb.Image(eval_dataset)})

        model = RHVAE(
            model_config=model_config
        )

        # trainer = BaseTrainer(
        #     model=model,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        #     training_config=training_config
        # )

        #trainer.train()

        pipeline = TrainingPipeline(
            training_config=training_config,
            model=model
        )


        pipeline(
            train_data=train_dataset,
            eval_data=eval_dataset,
            callbacks=callbacks # pass the callbacks to the TrainingPipeline and you are done!
        )

        # Generate synthetic data
        if args.sampler == 'normal':
            sampler = NormalSampler(
                model=model,
                sampler_config=None
            )
        elif args.sampler == 'gmm':
            sampler_config = GaussianMixtureSamplerConfig(
                n_components=10
            )

            sampler = GaussianMixtureSampler(
                sampler_config=sampler_config,
                model=model
            )

            sampler.fit(
                train_data=train_dataset
            )
        elif args.sampler == 'rhvae':
            sampler_config = RHVAESamplerConfig(
            )

            sampler = RHVAESampler(
                sampler_config=None,
                model=model
            )

            sampler.fit(
                train_data=train_dataset
            )

        gen_data = sampler.sample(
            num_samples=args.k,
        )

        # Update Training Dataset with Generated Data
        #train_dataset = ConcatDataset([train_dataset, gen_data])
        train_dataset = torch.cat((train_dataset, gen_data), 0)
