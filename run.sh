export OMP_NUM_THREADS=8
srun -p csc413 -c 8 --gres gpu:2 torchrun --nproc_per_node 2  manager.py train --wandb_key   --fresh_start