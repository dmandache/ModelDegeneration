import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets
from pythae.data.datasets import DatasetOutput
from pythae.models import *
from pythae.trainers import BaseTrainerConfig, BaseTrainer
from pythae.pipelines.training import TrainingPipeline
from pythae.trainers.training_callbacks import WandbCallback
from pythae.samplers import *
from pythae.models.nn.benchmarks.mnist import *
from pythae.models.nn.default_architectures import *
import wandb
import argparse
import random
from datetime import datetime
import os

## Group runs by by experiment
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.environ["WANDB_RUN_GROUP"] = f"experiment_{timestamp}"

## Args dictionary
model_dict = {
    'vae': VAE,
    'rhvae': RHVAE
    }

architecture_dict = {
    'mlp':
        {
        'encoder': Encoder_VAE_MLP,
        'decoder': Decoder_AE_MLP,
        },
    'convnet':
        {
        'encoder': Encoder_Conv_VAE_MNIST,
        'decoder': Decoder_Conv_AE_MNIST,
        },
    'resnet':
        {
        'encoder': Encoder_ResNet_VAE_MNIST,
        'decoder': Decoder_ResNet_AE_MNIST,
        },
    }


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

    
if __name__ == '__main__':

    # Argument Parser
    parser = argparse.ArgumentParser() #description='Train a RHVAE with synthetic data generation.')
    parser.add_argument('--input_dim', type=int, default=28, help='Dimensionality of the input data')
    parser.add_argument('--latent_dim', type=int, default=2, help='Dimensionality of the latent space')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of degenerating runs')
    parser.add_argument('--n_train', type=int, default=20, help='Number of training samples per class')
    parser.add_argument('--n_test', type=int, default=20, help='Number of test samples per class')
    parser.add_argument('--k', type=int, default=200, help='Number of synthetic data samples to generate at each iteration')
    parser.add_argument('--sampler', choices=['normal', 'gmm', 'rhvae'], default='rhvae', help='Sampler type for generating synthetic data')
    parser.add_argument('--architecture', choices=['convnet','resnet', 'mlp'], default='mlp', help='Model Architecture')
    parser.add_argument('--model', choices=['rhvae','vae'], default='rhvae', help='VAE Model')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs for each run')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model Config
    if args.model == 'rhvae':
        model_config = RHVAEConfig(
            input_dim=(1, args.input_dim, args.input_dim),
            latent_dim=args.latent_dim,
            # n_lf=1,
            # eps_lf=0.001,
            # beta_zero=0.3,
            # temperature=1.5,
            # regularization=0.001
            n_lf=3,
            eps_lf=0.001,
            beta_zero=0.3,
            temperature=0.8,
            regularization=0.01
        )
    elif args.model == 'vae':
        model_config = VAEConfig(
            input_dim=(1, args.input_dim, args.input_dim),
            latent_dim=args.latent_dim
        )

    # Training Config
    training_config = BaseTrainerConfig(
        output_dir=f'experiments/{timestamp}',
        num_epochs=args.n_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 5, "factor": 0.5}
    )

    # Real data loader
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    
    train_indeces = sample_indices(mnist_trainset.targets, k=args.n_train, seed=42)
    remaining_indeces = list(set(range(len(mnist_trainset.targets)))-set(train_indeces))
    eval_indeces = sample_indices(mnist_trainset.targets[remaining_indeces], k=args.n_test, seed=42)

    train_dataset = mnist_trainset.data[train_indeces].reshape(-1, 1, 28, 28) / 255.
    eval_dataset = mnist_trainset.data[eval_indeces].reshape(-1, 1, 28, 28) / 255.
    print(train_dataset.shape, eval_dataset.shape)

    train_dataset = train_dataset.to(device)
    eval_dataset = eval_dataset.to(device)


    # Training loop
    for i in range(args.n_runs):

        print(f"RUN {i}")

        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config=training_config, # training config
            model_config=model_config, # model config
            project_name="model-degeneration", # specify your wandb project,
        )

        wandb.run.name = f"experiment_{timestamp}_run_{i}"
        wandb.config.update(args)

        callbacks = []
        callbacks.append(wandb_cb)

        if i==0:
            wandb.log({"Training Data": wandb.Image(train_dataset),
                       "Evaluation Data": wandb.Image(eval_dataset)})
        else:
            wandb.log({"Generated Data": wandb.Image(gen_data)})


        model = model_dict[args.model](
            model_config=model_config,
            encoder=architecture_dict[args.architecture]['encoder'](model_config), 
            decoder=architecture_dict[args.architecture]['decoder'](model_config) 
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
                mcmc_steps_nbr=100,
                n_lf=10,
                eps_lf=0.03
            )

            sampler = RHVAESampler(
                sampler_config=sampler_config,
                model=model
            )

            sampler.fit(
                train_data=train_dataset
            )

        gen_data = sampler.sample(
            num_samples=args.k,
        )

        gen_data = gen_data.to(device)

        # Update Training Dataset with Generated Data
        #train_dataset = ConcatDataset([train_dataset, gen_data])
        train_dataset = torch.cat((train_dataset, gen_data), 0)
        # shuffle
        train_dataset = train_dataset[torch.randperm(train_dataset.size()[0])]
