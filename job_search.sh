#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=92G
#SBATCH --partition csc413
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=8
torchrun --nproc_per_node 1  manager.py train --wandb  --fresh_start --tiny --epochs 15 --search