#!/bin/bash
# Set Slurm options
#SBATCH --job-name=model-degeneration
#SBATCH --output=model-degeneration.out
#SBATCH --error=model-degeneration.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB

# Load your environment (conda, ...)
module load cuda
source /home/$USER/.bashrc
conda activate py310

python main.py --input_dim 28 --latent_dim 8 --n_train 1000 --n_test 200 --n_runs 5 --n_epochs 1000 --k 10000 --model rhvae --architecture convnet --sampler rhvae 
