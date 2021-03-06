#!/bin/bash

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation_fliploss_1/UPDRS/Kinect_3d

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation_fliploss_1/SAS/Kinect_3d

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh
