#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=92G
#SBATCH --partition csc413
#SBATCH --gres=gpu:2

export OMP_NUM_THREADS=8
python faithfulness_calculation.py