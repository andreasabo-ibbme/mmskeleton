#!/bin/bash
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=8000               # memory (per node)
#SBATCH --cpus-per-task=8
#SBATCH --time=0-04:00            # time (DD-HH:MM)
#SBATCH --job-name=v2_nn_simpl10_sas_120
#SBATCH --output=%x-%j_32hour.out
#SBATCH --account=def-btaati

#SBATCH --mail-user=andrea.sabo@mail.utoronto.ca

#SBATCH --mail-type=FAIL



module load python/3.6
module load nixpkgs/16.09  gcc/8.3.0 
module load cuda/10.1

source ~/ENV/bin/activate

cd /home/asabo/projects/def-btaati/asabo/mmskeleton
python mmskl.py configs/recognition/tri/cluster/v2/SAS/PD_only/train_cluster_SAS_10_120_nonorm.yaml
