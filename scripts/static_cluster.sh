#!/bin/bash

# export CUDA_HOME=/is/software/nvidia/cuda-11.8
# export LD_LIBRARY_PATH=/is/software/nvidia/cuda-11.8/lib64
# export PATH=$PATH:/is/software/nvidia/cuda-11.8/bin

if command -v nvcc &> /dev/null; then
    CUDA_HOME=$(dirname $(dirname $(which nvcc)))
elif [ -d "/is/software/nvidia/cuda-11.8" ]; then
    CUDA_HOME=/is/software/nvidia/cuda-11.8
else
    echo "CUDA not found! Please install or add nvcc to PATH."
    return 1  # works if sourced
fi

export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA detected at: $CUDA_HOME"
echo "LD_LIBRARY_PATH detected at: $LD_LIBRARY_PATH"


export GPU="0"
export CUDA_VISIBLE_DEVICES=$GPU


# Activate conda environment
if [ -f "$HOME/miniconda_latest3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda_latest3/etc/profile.d/conda.sh"
    PYTHON_ENV="$HOME/miniconda_latest3/bin/activate"
    source "$PYTHON_ENV"
    
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    PYTHON_ENV="$HOME/anaconda3/bin/activate"
    source "$PYTHON_ENV"
    
else
    echo "Conda not found in standard locations!"
fi


conda activate /home/vsklyarova/miniconda_latest3/envs/eccv_gaus_hair 

PORT=$((1000 + RANDOM % 5001))

echo "Starting training ..."
echo ""

export PYTHONPATH=./submodules/external/VOODOO3D-official:$PYTHONPATH
export PYTHONPATH=./submodules/external/GaussianHaircut:$PYTHONPATH

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