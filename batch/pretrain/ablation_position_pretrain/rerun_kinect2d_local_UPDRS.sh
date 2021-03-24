#!/bin/bash
pyenv activate mmskeleton

cd /home/saboa/code/mmskeleton

# # UPDRS-gait

# Kinect 2D
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/kinect_2d/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.4.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/kinect_2d/gait_features/train_cluster_10_120_pred_15_4_joints_do_0.1.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/kinect_2d/gait_features/train_cluster_10_120_pred_15_4_joints_do_0.2.yaml

python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/kinect_2d/no_gait_features/train_cluster_2_120_pred_15_4_joints_do_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/kinect_2d/no_gait_features/train_cluster_2_120_pred_15_4_joints_do_0.3.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/kinect_2d/no_gait_features/train_cluster_2_120_pred_15_4_joints_do_0.4.yaml

