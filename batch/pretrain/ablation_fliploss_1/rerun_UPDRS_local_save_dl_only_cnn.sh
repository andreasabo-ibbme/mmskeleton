#!/bin/bash

pyenv activate mmskeleton

cd /home/saboa/code/mmskeleton


python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_cnn/UPDRS/alphapose/temp5/model_1/cnn_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_cnn/UPDRS/detectron/temp5/model_1/cnn_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_cnn/UPDRS/openpose/temp5/model_1/cnn_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_cnn/UPDRS/kinect_3d/temp5/model_1/cnn_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_cnn/UPDRS/kinect_2d/temp5/model_1/cnn_0.0.yaml
