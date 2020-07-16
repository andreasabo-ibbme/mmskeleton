#!/bin/bash

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/v2
cd ./SAS/v3
sbatch SAS_0_120_nonorm.sh
sbatch SAS_0_120_nonorm_do_0.1.sh
sbatch SAS_2_120_nonorm.sh
sbatch SAS_2_120_nonorm_do_0.1.sh


cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/v2
cd ./UPDRS/v3
sbatch SAS_0_120_nonorm.sh
sbatch SAS_0_120_nonorm_do_0.1.sh
sbatch SAS_2_120_nonorm.sh
sbatch SAS_2_120_nonorm_do_0.1.sh