#!/bin/bash

pyenv activate mmskeleton

cd /home/saboa/code/mmskeleton


python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1_xavier_no_pretrain/SAS/alphapose/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.1.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_flip_1_xavier_no_pretrain/SAS/kinect_3d/gait_features/train_cluster_2_120_pred_15_4_joints_do_0.0.yaml


