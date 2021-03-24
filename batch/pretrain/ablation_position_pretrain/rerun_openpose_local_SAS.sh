#!/bin/bash
pyenv activate mmskeleton

cd /home/saboa/code/mmskeleton

# # UPDRS-gait

# openpose
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/SAS/openpose/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.3.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/SAS/openpose/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.4.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/SAS/openpose/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.5.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/SAS/openpose/no_gait_features/train_cluster_2_120_pred_15_4_joints_do_0.3.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/SAS/openpose/no_gait_features/train_cluster_2_120_pred_15_4_joints_do_0.4.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/SAS/openpose/no_gait_features/train_cluster_2_120_pred_15_4_joints_do_0.5.yaml