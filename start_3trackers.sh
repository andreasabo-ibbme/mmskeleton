#!/bin/bash

pyenv activate mmskeleton

cd /home/saboa/code/mmskeleton

python mmskl.py configs/recognition/tri/cluster/3_tracker/v1/UPDRS/alphapose/train_cluster_0_120_pred_15_4_joints_do_0.2.yaml
python mmskl.py configs/recognition/tri/cluster/3_tracker/v1/UPDRS/alphapose/train_cluster_0_120_pred_15_4_joints_do_0.4.yaml
python mmskl.py configs/recognition/tri/cluster/3_tracker/v1/UPDRS/detectron/train_cluster_0_120_pred_15_4_joints_do_0.1.yaml
python mmskl.py configs/recognition/tri/cluster/3_tracker/v1/UPDRS/detectron/train_cluster_0_120_pred_15_4_joints_do_0.2.yaml
python mmskl.py configs/recognition/tri/cluster/3_tracker/v1/UPDRS/detectron/train_cluster_2_120_pred_15_4_joints_do_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/3_tracker/v1/UPDRS/alphapose/train_cluster_0_120_pred_15_4_joints_do_0.0.yaml
