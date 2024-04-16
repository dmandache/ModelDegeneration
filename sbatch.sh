#!/bin/bash
# Set Slurm options
#SBATCH --job-name=model-degeneration
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=48:00:00     # Time limit hrs:min:sec

# Load your environment (conda, ...)
module load cuda
source /home/$USER/.bashrc
conda activate py310

python main.py --input_dim 28 --latent_dim 8 --n_train 200 --n_test 50 --n_runs 3 --batch_size -1 --n_epochs 500 --k 2000 --model rhvae --architecture tiny --sampler rhvae --loss bce