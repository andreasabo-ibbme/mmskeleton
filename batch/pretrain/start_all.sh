#!/bin/bash

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain

# UDPRS
cd ./UPDRS/
sbatch UPDRS_2_120_pred_15,30.sh
sbatch UPDRS_2_120_pred_30.sh
sbatch UPDRS_10_120_pred_15,30.sh
sbatch UPDRS_10_120_pred_30.sh
sbatch UPDRS_11_120_pred_15,30.sh
sbatch UPDRS_11_120_pred_30.sh

# SAS
cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain
cd ./SAS/
sbatch SAS_2_120_pred_15,30.sh
sbatch SAS_2_120_pred_30.sh
sbatch SAS_10_120_pred_15,30.sh
sbatch SAS_10_120_pred_30.sh
sbatch SAS_11_120_pred_15,30.sh
sbatch SAS_11_120_pred_30.sh
