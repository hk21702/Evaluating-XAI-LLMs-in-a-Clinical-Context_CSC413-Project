#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=92G
#SBATCH --partition csc413
#SBATCH --gres=gpu:2

export OMP_NUM_THREADS=8
torchrun --nproc_per_node 2  manager.py train --project_name OPT-Finetuning-ICD10 --train_path data/train_10.csv --val_path data/val_10.csv --test_path data/test_10.csv --code_labels data/icd10_codes.csv --checkpoint_dir opt-finetuned-icd10 --wandb_key 971c33204c93100c21249224f1d7e3082f3d2ad7 --fresh_start 
