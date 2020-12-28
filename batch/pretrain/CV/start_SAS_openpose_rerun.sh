#!/bin/bash

# Rerun the SAS configs that failed
cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/CV/SAS/openpose

sbatch gait_features/config_2.sh
sbatch gait_features/config_5.sh

sbatch no_gait_features/config_4.sh