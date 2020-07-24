#!/bin/bash
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=8000               # memory (per node)
#SBATCH --cpus-per-task=8
#SBATCH --time=0-24:00            # time (DD-HH:MM)
#SBATCH --job-name=SAS_2_120_pred_15
#SBATCH --output=%x-%j_32hour.out
#SBATCH --account=def-btaati

#SBATCH --mail-user=andrea.sabo@mail.utoronto.ca
#SBATCH --mail-type=FAIL


module load python/3.6
module load nixpkgs/16.09  gcc/8.3.0 
module load cuda/10.1

source ~/ENV/bin/activate

cd /home/asabo/projects/def-btaati/asabo/mmskeleton
python mmskl.py configs/recognition/tri/cluster/pred_pretrain_coco_simplified_head_ankles_ankle_wrists/SAS/v2/train_cluster_0_120_pred_15,30_4_joints.yaml