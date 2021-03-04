#!/bin/bash

pyenv activate mmskeleton

cd /home/saboa/code/mmskeleton

# DONE
# python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1/UPDRS/alphapose/gait_features/train_cluster_0_120_pred_15_4_joints_do_0.2_temp9.yaml

#UPDRS gait fts
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1/UPDRS/openpose/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.1.yaml

#SAS no gait fts
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1/SAS/openpose/no_gait_features/train_cluster_2_120_pred_15_4_joints_do_0.3.yaml