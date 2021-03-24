#!/bin/bash
pyenv activate mmskeleton

cd /home/saboa/code/mmskeleton

# # UPDRS-gait

# alphapose
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/alphapose/gait_features/train_cluster_0_120_pred_15_4_joints_do_0.2_temp9.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/alphapose/gait_features/train_cluster_0_120_pred_15_4_joints_do_0.2_temp13.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/alphapose/gait_features/train_cluster_0_120_pred_15_4_joints_do_0.3.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/alphapose/no_gait_features/train_cluster_0_120_pred_15_4_joints_do_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/alphapose/no_gait_features/train_cluster_0_120_pred_15_4_joints_do_0.2.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/alphapose/no_gait_features/train_cluster_0_120_pred_15_4_joints_do_0.4.yaml
