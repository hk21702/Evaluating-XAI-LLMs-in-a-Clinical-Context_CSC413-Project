#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=92G
#SBATCH --partition csc413
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=8
torchrun bert_shap.py