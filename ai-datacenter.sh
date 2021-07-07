module purge

module load cuda/11.1

srun --gres=gpu:4 --pty bash -i