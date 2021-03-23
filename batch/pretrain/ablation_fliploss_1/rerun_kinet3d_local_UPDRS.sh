#!/bin/bash
pyenv activate mmskeleton

cd /home/saboa/code/mmskeleton

# UPDRS-gait
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1/UPDRS/kinect_3d/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1/UPDRS/kinect_3d/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.1.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1/UPDRS/kinect_3d/gait_features/train_cluster_10_120_pred_15_4_joints_do_0.0.yaml


# # SAS-gait
# python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1/SAS/kinect_3d/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.0.yaml
# python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1/SAS/kinect_3d/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.1.yaml
# python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1/SAS/kinect_3d/gait_features/train_cluster_10_120_pred_15_4_joints_do_0.1.yaml
