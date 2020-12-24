#!/bin/bash

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/CV/SAS/Kinect_3d

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh
sbatch gait_features/config_4.sh
sbatch gait_features/config_5.sh

sbatch no_gait_features/config_3.sh
sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_4.sh
sbatch no_gait_features/config_5.sh