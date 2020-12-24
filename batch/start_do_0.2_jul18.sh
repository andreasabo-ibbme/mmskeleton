#!/bin/bash

# Baselines
cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/v2
cd ./SAS/v3
sbatch SAS_0_120_nonorm_do_0.2.sh
sbatch SAS_2_120_nonorm_do_0.2.sh

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/v2
cd ./UPDRS/v3
sbatch SAS_0_120_nonorm_do_0.2.sh
sbatch SAS_2_120_nonorm_do_0.2.sh


# Pretraining
cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain

# UPDRS
cd ./UPDRS/v2/
sbatch SAS_0_120_pred_15_mse_do_0.2.sh
sbatch SAS_0_120_pred_15_do_0.2.sh
sbatch SAS_2_120_pred_15_mse_do_0.2.sh
sbatch SAS_2_120_pred_15_do_0.2.sh

# SAS
cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain
cd ./SAS/v2/
sbatch SAS_0_120_pred_15_mse_do_0.2.sh
sbatch SAS_0_120_pred_15_do_0.2.sh
sbatch SAS_2_120_pred_15_mse_do_0.2.sh
sbatch SAS_2_120_pred_15_do_0.2.sh