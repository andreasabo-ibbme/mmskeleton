#!/bin/bash

# UPDRS
cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/supcon
cd ./UPDRS/
sbatch UPDRS_11_120_no_norm_128_stgcn.sh
sbatch UPDRS_11_120_no_norm_64_stgcn.sh
sbatch UPDRS_11_120_no_norm_128_mlp.sh
sbatch UPDRS_11_120_no_norm_64_mlp.sh
sbatch UPDRS_10_120_no_norm_128_stgcn.sh
sbatch UPDRS_10_120_no_norm_64_stgcn.sh
sbatch UPDRS_10_120_no_norm_128_mlp.sh
sbatch UPDRS_10_120_no_norm_64_mlp.sh
sbatch UPDRS_2_120_no_norm_128_stgcn.sh
sbatch UPDRS_2_120_no_norm_64_stgcn.sh
sbatch UPDRS_2_120_no_norm_128_mlp.sh
sbatch UPDRS_2_120_no_norm_64_mlp.sh

