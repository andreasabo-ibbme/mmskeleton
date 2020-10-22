#!/bin/bash

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/kinect/SAS/v1

# SAS
sbatch temp5/SAS_0_120_pred_15_do_0.4.sh
sbatch temp5/SAS_2_120_pred_15_do_0.4.sh
sbatch temp5/SAS_10_120_pred_15_do_0.4.sh

sbatch temp9/SAS_0_120_pred_15_do_0.4.sh
sbatch temp9/SAS_2_120_pred_15_do_0.4.sh
sbatch temp9/SAS_10_120_pred_15_do_0.4.sh

sbatch temp13/SAS_0_120_pred_15_do_0.4.sh
sbatch temp13/SAS_2_120_pred_15_do_0.4.sh
sbatch temp13/SAS_10_120_pred_15_do_0.4.sh
