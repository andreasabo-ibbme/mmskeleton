#!/bin/bash

pyenv activate mmskeleton

cd /home/saboa/code/mmskeleton

# 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1/UPDRS/alphapose/gait_features/train_cluster_0_120_pred_15_4_joints_do_0.2_temp9.yaml

# cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation_fliploss_1/UPDRS/alphapose

# sbatch gait_features/config_1.sh
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

# sbatch gait_features/config_1.sh
# sbatch gait_features/config_2.sh
# sbatch gait_features/config_3.sh

# sbatch no_gait_features/config_1.sh
# sbatch no_gait_features/config_2.sh
# sbatch no_gait_features/config_3.sh