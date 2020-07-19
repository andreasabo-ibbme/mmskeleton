source ~/.bashrc

pyenv activate mmskeleton
pip list
cd /home/saboa/code/mmskeleton

#UPDRS
python mmskl.py /home/saboa/code/mmskeleton/configs/recognition/tri/cluster/supcon/UPDRS/train_cluster_0_120_no_norm_64_feats_mlp.yaml
python mmskl.py /home/saboa/code/mmskeleton/configs/recognition/tri/cluster/supcon/UPDRS/train_cluster_0_120_no_norm_128_feats_mlp.yaml
python mmskl.py /home/saboa/code/mmskeleton/configs/recognition/tri/cluster/supcon/UPDRS/train_cluster_2_120_no_norm_64_feats_mlp.yaml
python mmskl.py /home/saboa/code/mmskeleton/configs/recognition/tri/cluster/supcon/UPDRS/train_cluster_2_120_no_norm_128_feats_mlp.yaml

# SAS
python mmskl.py /home/saboa/code/mmskeleton/configs/recognition/tri/cluster/supcon/UPDRS/train_cluster_0_120_no_norm_64_feats_mlp.yaml
python mmskl.py /home/saboa/code/mmskeleton/configs/recognition/tri/cluster/supcon/UPDRS/train_cluster_0_120_no_norm_128_feats_mlp.yaml
python mmskl.py /home/saboa/code/mmskeleton/configs/recognition/tri/cluster/supcon/UPDRS/train_cluster_2_120_no_norm_64_feats_mlp.yaml
python mmskl.py /home/saboa/code/mmskeleton/configs/recognition/tri/cluster/supcon/UPDRS/train_cluster_2_120_no_norm_128_feats_mlp.yaml

