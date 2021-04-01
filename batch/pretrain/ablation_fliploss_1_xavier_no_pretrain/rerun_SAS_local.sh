#!/bin/bash

pyenv activate mmskeleton

cd /home/saboa/code/mmskeleton

# DONE
# python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1_xavier_no_pretrain/UPDRS/alphapose/gait_features/train_cluster_0_120_pred_15_4_joints_do_0.2_temp9.yaml


# python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1_xavier_no_pretrain/UPDRS/openpose/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.1.yaml

# python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1_xavier_no_pretrain/UPDRS/kinect_3d/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1_xavier_no_pretrain/SAS/kinect_3d/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.0.yaml
# cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation_fliploss_1/UPDRS/alphapose

# sbatch gait_features/config_1.sh #DONE
# sbatch gait_features/config_2.sh
# sbatch gait_features/config_3.sh


# cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation_fliploss_1/UPDRS/openpose

# sbatch gait_features/config_1.sh
# sbatch gait_features/config_2.sh
# sbatch gait_features/config_3.sh


# cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation_fliploss_1/UPDRS/detectron

# sbatch gait_features/config_1.sh
# sbatch gait_features/config_2.sh
# sbatch gait_features/config_3.sh


# cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation_fliploss_1/UPDRS/Kinect_3d

# sbatch no_gait_features/config_1.sh
# sbatch no_gait_features/config_2.sh
# sbatch no_gait_features/config_3.sh