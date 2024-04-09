#!/bin/bash

# Set Slurm options
#SBATCH --job-name=run_slurm_jobs
#SBATCH --output=run_slurm_jobs.out
#SBATCH --error=run_slurm_jobs.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB

# Loop through all Slurm job scripts and submit them
for script in slurm_scripts/*.sh; do
    sbatch "$script"
done
