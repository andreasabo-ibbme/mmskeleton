#!/bin/bash
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate mmskeleton
cd /home/saboa/code/mmskeleton

python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp5/train_cluster_0_120_pred_15_4_joints_do_0.0.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp5/train_cluster_2_120_pred_15_4_joints_do_0.0.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp5/train_cluster_10_120_pred_15_4_joints_do_0.0.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp9/train_cluster_0_120_pred_15_4_joints_do_0.0.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp9/train_cluster_2_120_pred_15_4_joints_do_0.0.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp9/train_cluster_10_120_pred_15_4_joints_do_0.0.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp13/train_cluster_0_120_pred_15_4_joints_do_0.0.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp13/train_cluster_2_120_pred_15_4_joints_do_0.0.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp13/train_cluster_10_120_pred_15_4_joints_do_0.0.yaml 

python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp5/train_cluster_0_120_pred_15_4_joints_do_0.1.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp5/train_cluster_2_120_pred_15_4_joints_do_0.1.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp5/train_cluster_10_120_pred_15_4_joints_do_0.1.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp9/train_cluster_0_120_pred_15_4_joints_do_0.1.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp9/train_cluster_2_120_pred_15_4_joints_do_0.1.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp9/train_cluster_10_120_pred_15_4_joints_do_0.1.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp13/train_cluster_0_120_pred_15_4_joints_do_0.1.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp13/train_cluster_2_120_pred_15_4_joints_do_0.1.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp13/train_cluster_10_120_pred_15_4_joints_do_0.1.yaml 

python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp5/train_cluster_0_120_pred_15_4_joints_do_0.2.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp5/train_cluster_2_120_pred_15_4_joints_do_0.2.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp5/train_cluster_10_120_pred_15_4_joints_do_0.2.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp9/train_cluster_0_120_pred_15_4_joints_do_0.2.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp9/train_cluster_2_120_pred_15_4_joints_do_0.2.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp9/train_cluster_10_120_pred_15_4_joints_do_0.2.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp13/train_cluster_0_120_pred_15_4_joints_do_0.2.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp13/train_cluster_2_120_pred_15_4_joints_do_0.2.yaml 
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/SAS/v6/temp13/train_cluster_10_120_pred_15_4_joints_do_0.2.yaml 