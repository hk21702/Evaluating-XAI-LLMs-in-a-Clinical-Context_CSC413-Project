#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=92G
#SBATCH --partition csc413
#SBATCH --gres=gpu:2

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
torchrun --nproc_per_node 2  manager.py train --wandb --fresh_start --project_name "OPT Finetuning 350m mimic s10 e10"  --checkpoint_dir "OPT-350m-mimic-full s10 e10" --gradient_checkpointing --epoch 15 --tiny --shard 10