#!/bin/bash

pyenv activate mmskeleton

cd /home/saboa/code/mmskeleton


python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_cnn/SAS/alphapose/temp5/model_1/cnn_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_cnn/SAS/detectron/temp5/model_1/cnn_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_cnn/SAS/openpose/temp5/model_1/cnn_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_cnn/SAS/kinect_3d/temp5/model_1/cnn_0.0.yaml
python mmskl.py configs/recognition/tri/cluster/pred_pretrain/ablation_configs_cnn/SAS/kinect_2d/temp5/model_1/cnn_0.0.yaml
