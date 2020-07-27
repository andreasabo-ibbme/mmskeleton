cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/jul22_scale_ablation/pretrain/SAS/v3

## STGCN

cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/jul22_scale_ablation/stgcn/SAS/v3

# SAS 
sbatch SAS_0_120_pred_15_ft_all.sh
sbatch SAS_0_120_pred_15_do_0.1_ft_all.sh

sbatch SAS_2_120_pred_15_ft_all.sh
sbatch SAS_2_120_pred_15_do_0.1_ft_all.sh

sbatch SAS_10_120_pred_15_ft_all.sh
sbatch SAS_10_120_pred_15_do_0.1_ft_all.sh

# UPDRS 
cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/jul22_scale_ablation/stgcn/UPDRS/v3
sbatch SAS_0_120_pred_15_ft_all.sh
sbatch SAS_0_120_pred_15_do_0.1_ft_all.sh

sbatch SAS_2_120_pred_15_ft_all.sh
sbatch SAS_2_120_pred_15_do_0.1_ft_all.sh

sbatch SAS_10_120_pred_15_ft_all.sh
sbatch SAS_10_120_pred_15_do_0.1_ft_all.sh


cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/jul22_scale_ablation/stgcn_coco_simplified_head_ankles_ankle_wrists/SAS/v3

# SAS pretrain_coco_simplified_head_ankles_ankle_wrists
sbatch SAS_0_120_pred_15_ft_all.sh
sbatch SAS_0_120_pred_15_do_0.1_ft_all.sh

sbatch SAS_2_120_pred_15_ft_all.sh
sbatch SAS_2_120_pred_15_do_0.1_ft_all.sh

sbatch SAS_10_120_pred_15_ft_all.sh
sbatch SAS_10_120_pred_15_do_0.1_ft_all.sh

# UPDRS pretrain_coco_simplified_head_ankles_ankle_wrists
cd /home/asabo/projects/def-btaati/asabo/mmskeleton/batch/jul22_scale_ablation/stgcn_coco_simplified_head_ankles_ankle_wrists/UPDRS/v3
sbatch SAS_0_120_pred_15_ft_all.sh
sbatch SAS_0_120_pred_15_do_0.1_ft_all.sh

sbatch SAS_2_120_pred_15_ft_all.sh
sbatch SAS_2_120_pred_15_do_0.1_ft_all.sh

sbatch SAS_10_120_pred_15_ft_all.sh
sbatch SAS_10_120_pred_15_do_0.1_ft_all.sh