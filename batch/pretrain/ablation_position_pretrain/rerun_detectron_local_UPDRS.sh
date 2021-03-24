#!/bin/bash
pyenv activate mmskeleton

cd /home/saboa/code/mmskeleton

# # UPDRS-gait

# detectron
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/detectron/gait_features/train_cluster_0_120_pred_15_4_joints_do_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/detectron/gait_features/train_cluster_0_120_pred_15_4_joints_do_0.2.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/detectron/gait_features/train_cluster_0_120_pred_15_4_joints_do_0.3.yaml

python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/detectron/no_gait_features/train_cluster_0_120_pred_15_4_joints_do_0.1.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/detectron/no_gait_features/train_cluster_0_120_pred_15_4_joints_do_0.2.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_pretrain_position_err/UPDRS/detectron/no_gait_features/train_cluster_2_120_pred_15_4_joints_do_0.0.yaml