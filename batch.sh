#!/bin/bash
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=8000               # memory (per node)
#SBATCH --time=0-01:40            # time (DD-HH:MM)
#SBATCH --job-name=cedar_mmskel
#SBATCH --output=%x-%j.out



module load python/3.6
module load nixpkgs/16.09  gcc/8.3.0 
module load cuda/10.1

source ~/ENV/bin/activate

cd /home/asabo/projects/def-btaati/asabo/mmskeleton
python mmskl.py configs/recognition/tri/train_cluster.yaml


