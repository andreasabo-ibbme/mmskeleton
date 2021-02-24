#!/bin/bash

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation/UPDRS/alphapose

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation/UPDRS/openpose

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation/UPDRS/detectron

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation/UPDRS/Kinect_2d

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/ablation/UPDRS/Kinect_3d

sbatch gait_features/config_1.sh
sbatch gait_features/config_2.sh
sbatch gait_features/config_3.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_2.sh
sbatch no_gait_features/config_3.sh