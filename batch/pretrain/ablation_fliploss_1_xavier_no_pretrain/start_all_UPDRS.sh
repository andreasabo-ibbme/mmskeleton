#!/bin/bash

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation_fliploss_1/UPDRS/alphapose

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation_fliploss_1/UPDRS/openpose

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation_fliploss_1/UPDRS/detectron

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation_fliploss_1/UPDRS/Kinect_2d

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation_fliploss_1/UPDRS/Kinect3d

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh