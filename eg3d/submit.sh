CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--outdir=/storage/guangrun/qijia_3d_model/eg3d/imagenet_sub_seg128_whitebg \
--cfg=shapenet \
--gamma=0.3 \
--data=/datasets/guangrun/qijia_3d_model/imagenet/imagenet_sub_seg128_whitebg.zip \
--gpus=8 \
--batch=64 \
--gen_pose_cond=True \
--neural_rendering_resolution_initial=128 \
--metrics=None \
--cls_weight=8 \
--aug=ada