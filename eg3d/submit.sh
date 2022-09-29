#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --gres=gpu:4 --constraint='gpu_mem:32GB'

nvidia-smi
module load CUDA/11.4.1-GCC-10.3.0
module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate sgxl
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--outdir=/data/engs-tvg/engs2305/qijia_3d_model/storage/eg3d/training-runs \
--cfg=shapenet \
--gamma=0.3 \
--data=/data/engs-tvg/engs2305/qijia_3d_model/data/imagenet_sub_seg128_whitebg.zip \
--gpus=1 \
--batch=4 \
--gen_pose_cond=True \
--neural_rendering_resolution_initial=128 \
--metrics=None \
--cls_weight=8 \
--aug=ada