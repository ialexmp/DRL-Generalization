#!/bin/bash
#SBATCH -J inf0_0001
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=14-0:00
#SBATCH -o ../slurm-outputs/LearningRates/inf_lr_0_0001_results/%N.%J.out # STDOUT
#SBATCH -e ../slurm-outputs/LearningRates/inf_lr_0_0001_results/%N.%j.err # STDERR

ml Python
module load CUDA/11.4.3
source ../../venv/bin/activate
python ../../Algorithms/PPO_PickAndPlace/train_PaP.py inf_lr0_0001.yaml

deactivate
            
