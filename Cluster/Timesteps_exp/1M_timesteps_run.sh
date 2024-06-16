#!/bin/bash
#SBATCH -J 1M_timesteps
#SBATCH -p medium
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o ../slurm-outputs/Timestamps/%N.%J.out # STDOUT
#SBATCH -e ../slurm-outputs/Timestamps/%N.%j.err # STDERR

ml Python
module load CUDA/11.4.3
source ../../venv/bin/activate
python ../../Algorithms/PPO_PickAndPlace/train_PaP.py 1M_timesteps.yaml

deactivate
            
