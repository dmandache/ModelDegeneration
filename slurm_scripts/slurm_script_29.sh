#!/bin/bash
#SBATCH --job-name=model_degeneration
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB

# Load your environment (conda, ...)
source /home/$USER/.bashrc
conda activate py310

# Run Python script with arguments
python main.py --input_dim 28 --latent_dim 16 --n_train 1000 --n_test 100 --n_runs 10 --n_epochs 100 --k 1000 --sampler rhvae --architecture convnet
