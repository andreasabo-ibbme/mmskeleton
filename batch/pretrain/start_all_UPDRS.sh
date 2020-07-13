#!/bin/bash

# UPDRS
cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/pretrain
cd ./UPDRS/
sbatch UPDRS_2_120_pred_15,30.sh
sbatch UPDRS_2_120_pred_30.sh
sbatch UPDRS_10_120_pred_15,30.sh
sbatch UPDRS_10_120_pred_30.sh
sbatch UPDRS_11_120_pred_15,30.sh
sbatch UPDRS_11_120_pred_30.sh
