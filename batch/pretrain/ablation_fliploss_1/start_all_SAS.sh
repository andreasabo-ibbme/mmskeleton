#!/bin/bash

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation/SAS/alphapose

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation/SAS/openpose

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation/SAS/detectron

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation/SAS/Kinect_2d

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation/SAS/Kinect_2d

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh