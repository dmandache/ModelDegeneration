#!/bin/bash
#
# array.sbatch
#
# Allocated resources are NOT SHARED across the jobs.
# They represent resources allocated for each job
# in the array.

#SBATCH --job-name=array_ex
#SBATCH --output=%x_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# Load your environment (conda, ...)
source /home/$USER/.bashrc
conda activate py310

python main.py f_${SLURM_ARRAY_TASK_ID}.in