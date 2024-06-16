#!/bin/bash
#SBATCH -J 25M_steps
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=37:00:00
#SBATCH -o ../slurm-outputs/Timesteps/25M_timesteps/%N.%J.out # STDOUT
#SBATCH -e ../slurm-outputs/Timesteps/25M_timesteps/%N.%j.err # STDERR

ml Python
module load CUDA/11.4.3
source ../../venv/bin/activate
python ../../Algorithms/PPO_PickAndPlace/train_PaP.py 25M_timesteps.yaml

deactivate
            
