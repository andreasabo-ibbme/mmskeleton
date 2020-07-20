#!/bin/bash
cd /home/asabo/projects/def-btaati/asabo/mmskeleton

sbatch batch/pretrain/UPDRS/v2/SAS_2_120_pred_15,30_do_0.2.sh
sbatch batch/pretrain/UPDRS/v2/SAS_2_120_pred_15_mse_do_0.2.sh
sbatch batch/pretrain/SAS/v2/SAS_2_120_pred_15,30_do_0.1.sh
sbatch batch/pretrain/SAS/v2/SAS_2_120_pred_15,30_do_0.2.sh
sbatch batch/v2/UPDRS/v3/SAS_0_120_nonorm_do_0.2.sh
sbatch batch/v2/SAS/v3/SAS_0_120_nonorm_do_0.2.sh
sbatch batch/supcon/UPDRS/UPDRS_0_120_no_norm_64_stgcn.sh
sbatch batch/supcon/UPDRS/UPDRS_0_120_no_norm_128_stgcn.sh

