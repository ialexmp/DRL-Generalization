#!/bin/bash
#SBATCH -J 10M_steps
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=17:00:00
#SBATCH -o ../slurm-outputs/Timesteps/10M_timesteps/%N.%J.out # STDOUT
#SBATCH -e ../slurm-outputs/Timesteps/10M_timesteps/%N.%j.err # STDERR

ml Python
module load CUDA/11.4.3
source ../../venv/bin/activate
python ../../Algorithms/PPO_PickAndPlace/train_PaP.py 10M_timesteps.yaml

deactivate
            
