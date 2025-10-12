#!/bin/bash

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin

export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA detected at: $CUDA_HOME"
echo "LD_LIBRARY_PATH detected at: $LD_LIBRARY_PATH"


export GPU="0"
export CUDA_VISIBLE_DEVICES=$GPU

source ~/anaconda3/bin/activate
conda activate clip

PORT=$((1000 + RANDOM % 5001))

echo "Starting training ..."
echo ""

export PYTHONPATH=./submodules/external/VOODOO3D-official:$PYTHONPATH
export PYTHONPATH=/localhome/aha220/Hairdar/modules/GaussianHaircut

conf_path="static.conf"
exp_name="try"
data_path="./data"
folder_name="examples"
RES="256"
NSTEPS="20"


for scene in $data_path/$folder_name/img/*; do
    scene_name=$(basename "$scene")  # extract filename only
    echo "Processing $scene_name..."

    python run_image_reconstruction.py \
        --conf_path ./configs/$conf_path \
        --savedir ./exps_inverse_stage/$exp_name/$folder_name/$scene_name \
        --unfreeze_time_for_pca -1 \
        --num_workers 1 \
        --ckpt_path "./pretrained_models/fine.pth" \
        -r 1 \
        --pointcloud_path_head "./data/pointcloud.ply" \
        --render_direction \
        --binarize_masks \
        --detect_anomaly \
        --port $PORT \
        --ip 127.0.0.13 \
        --scene "$scene_name" \
        --upsample_hairstyle True \
        --upsample_resolution $RES \
        --num_steps_coarse $NSTEPS \
        --folder_name $folder_name
done