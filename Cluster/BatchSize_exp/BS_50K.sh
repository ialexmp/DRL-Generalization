#!/bin/bash
#SBATCH -J BS50K
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=14-0:00
#SBATCH -o ../slurm-outputs/BatchSize/BS_50K/%N.%J.out # STDOUT
#SBATCH -e ../slurm-outputs/BatchSize/BS_50K/%N.%j.err # STDERR

ml Python
module load CUDA/11.4.3
source ../../venv/bin/activate
python ../../Algorithms/PPO_PickAndPlace/train_PaP.py inf_batch_50K.yaml

deactivate
            