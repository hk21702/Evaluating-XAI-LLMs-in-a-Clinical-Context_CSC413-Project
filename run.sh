export OMP_NUM_THREADS=8
srun -p csc413 --mem=92G --gres gpu:2 torchrun --nproc_per_node 2  manager.py train --wandb --fresh_start --tiny --epochs 3