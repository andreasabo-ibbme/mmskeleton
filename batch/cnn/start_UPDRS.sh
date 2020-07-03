#!/bin/bash

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/cnn
cd ./UPDRS/

sbatch UPDRS_3.sh
sbatch UPDRS_3_no_bn.sh
sbatch UPDRS_3_no_bn_do_0.1.sh
sbatch UPDRS_3_do_0.1.sh
sbatch UPDRS_3_120.sh
sbatch UPDRS_3_120_no_bn.sh
sbatch UPDRS_3_120_no_bn_do_0.1.sh
sbatch UPDRS_3_120_do_0.1.sh

sbatch UPDRS_4.sh
sbatch UPDRS_4_no_bn.sh
sbatch UPDRS_4_no_bn_do_0.1.sh
sbatch UPDRS_4_do_0.1.sh
sbatch UPDRS_4_120.sh
sbatch UPDRS_4_120_no_bn.sh
sbatch UPDRS_4_120_no_bn_do_0.1.sh
sbatch UPDRS_4_120_do_0.1.sh
