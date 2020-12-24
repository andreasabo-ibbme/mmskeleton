#!/bin/bash

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain_coco_simplified_head_ankles_ankle_wrists

# UPDRS
cd ./UPDRS/v2/
sbatch SAS_0_120_pred_15.sh
sbatch SAS_0_120_pred_15_mse.sh
sbatch SAS_0_120_pred_15_mse_do_0.1.sh
sbatch SAS_0_120_pred_15_do_0.1.sh
sbatch SAS_2_120_pred_15.sh
sbatch SAS_2_120_pred_15_mse.sh
sbatch SAS_2_120_pred_15_mse_do_0.1.sh
sbatch SAS_2_120_pred_15_do_0.1.sh

# SAS
cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain_coco_simplified_head_ankles_ankle_wrists
cd ./SAS/v2/
sbatch SAS_0_120_pred_15.sh
sbatch SAS_0_120_pred_15_mse.sh
sbatch SAS_0_120_pred_15_mse_do_0.1.sh
sbatch SAS_0_120_pred_15_do_0.1.sh
sbatch SAS_2_120_pred_15.sh
sbatch SAS_2_120_pred_15_mse.sh
sbatch SAS_2_120_pred_15_mse_do_0.1.sh
sbatch SAS_2_120_pred_15_do_0.1.sh
