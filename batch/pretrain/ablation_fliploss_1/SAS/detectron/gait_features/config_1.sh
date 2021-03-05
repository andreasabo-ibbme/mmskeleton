#!/bin/bash
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=16000               # memory (per node)
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00            # time (DD-HH:MM)
#SBATCH --job-name=SAS_openpose_0.0
#SBATCH --output=%x-%j_32hour.out
#SBATCH --account=def-btaati

#SBATCH --mail-user=andrea.sabo@mail.utoronto.ca
#SBATCH --mail-type=FAIL


module load python/3.6
module load nixpkgs/16.09  gcc/8.3.0 
module load cuda/10.1

source ~/ENV/bin/activate

cd /home/asabo/projects/def-btaati/asabo/mmskeleton
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1/SAS/detectron/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.0.yaml