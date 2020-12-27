#!/bin/bash

# Rerun the SAS configs that failed
cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain/CV/SAS/detectron

sbatch gait_features/config_1.sh

sbatch no_gait_features/config_1.sh
sbatch no_gait_features/config_4.sh