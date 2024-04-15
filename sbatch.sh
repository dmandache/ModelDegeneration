#!/bin/bash
# Set Slurm options
#SBATCH --job-name=model-degeneration
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

python main.py --input_dim 28 --latent_dim 2 --n_train 50 --n_test 10 --n_runs 10 --batch_size -1 --n_epochs 100 --k 500 --model rhvae --architecture tiny --sampler rhvae 