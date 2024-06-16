#!/bin/bash
#SBATCH -J 10M001
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH -o ../slurm-outputs/LearningRates/10M_lr_0_001/%N.%J.out # STDOUT
#SBATCH -e ../slurm-outputs/LearningRates/10M_lr_0_001/%N.%j.err # STDERR

ml Python
module load CUDA/11.4.3
source ../../venv/bin/activate
python ../../Algorithms/PPO_PickAndPlace/train_PaP.py 10M_lr0_001.yaml

deactivate
            
