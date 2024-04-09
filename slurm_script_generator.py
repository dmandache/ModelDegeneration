import itertools
import os

# Define combinations of arguments
arguments = {
    'input_dim': [28],
    'latent_dim': [8, 16],
    'n_train' : [20, 100, 1000],
    'n_test' : [100],
    'n_runs': [10],
    'n_epochs': [100],
    'k': [200],
    'sampler': ['rhvae', 'normal', 'gmm'],
    'model': ['resnet', 'convnet']
}

# Generate all combinations of arguments
combinations = list(itertools.product(*arguments.values()))

# Directory to store Slurm job scripts
output_dir = './slurm_scripts'
os.makedirs(output_dir, exist_ok=True)

# Template for Slurm job script
slurm_template = """#!/bin/bash
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
python main.py --input_dim {input_dim} --latent_dim {latent_dim} --n_train {n_train} --n_test {n_test} --n_runs {n_runs} --n_epochs {n_epochs} --k {k} --sampler {sampler} --model {model}
"""

# Write Slurm job script for each combination of arguments
for i, combo in enumerate(combinations):
    script_content = slurm_template.format(input_dim=combo[0], hidden_dim=combo[1], latent_dim=combo[2],
                                            n_iters=combo[3], k=combo[4], sampler=combo[5])
    script_path = os.path.join(output_dir, f"slurm_script_{i}.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)

print(f"{len(combinations)} Slurm job scripts generated in {output_dir}/")
