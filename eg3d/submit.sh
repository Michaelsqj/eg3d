CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train.py \
--outdir=/storage/guangrun/qijia_3d_model/eg3d/shapenet_core_v2/training-runs \
--cfg=shapenet \
--data=/datasets/guangrun/qijia_3d_model/shapenet_core_v2_single_view_128.zip \
--gpus=4 \
--batch=32 \
--gamma=0.3 \
--gen_pose_cond=True \
--aug=ada

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train.py \
--outdir=/storage/guangrun/qijia_3d_model/eg3d/shapenet_core_v2/training-runs \
--cfg=shapenet \
--data=/datasets/guangrun/qijia_3d_model/shapenet_core_v2_single_view_128.zip \
--gpus=4 \
--batch=32 \
--gamma=0.3 \
--gen_pose_cond=True \
--aug=ada

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--outdir=/storage/guangrun/qijia_3d_model/eg3d/shapenet_core_v2/training-runs \
--cfg=shapenet \
--gamma=0.3 \
--data=/datasets/guangrun/qijia_3d_model/shapenet_core_v2_single_view_128.zip \
--gpus=8 \
--batch=64 \
--gamma=0.3 \
--gen_pose_cond=True \
--aug=ada \
--workers=8 \
--glr=0.0025 \
--dlr=0.002 \
--resume=/storage/guangrun/qijia_3d_model/eg3d/shapenet_core_v2/training-runs/00084-shapenet-shapenet_core_v2_single_view_128-gpus4-batch32-gamma0.3/network-snapshot-002000.pkl

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--outdir=/storage/guangrun/qijia_3d_model/eg3d/imagenet_sub_seg128_whitebg \
--cfg=shapenet \
--gamma=0.3 \
--data=/datasets/guangrun/qijia_3d_model/imagenet/imagenet_sub_seg128_whitebg.zip \
--gpus=8 \
--batch=32 \
--gamma=0.3 \
--gen_pose_cond=True \
--aug=ada \
--workers=8 \
--glr=0.0025 \
--dlr=0.002 \
--resume=/storage/guangrun/qijia_3d_model/eg3d/shapenet_core_v2/training-runs/00091-shapenet-shapenet_core_v2_single_view_128-gpus8-batch64/network-snapshot-000604.pkl