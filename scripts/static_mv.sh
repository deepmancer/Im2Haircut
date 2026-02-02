#!/bin/bash
# Multi-view 3DGS inference script
# Processes each subject using all available views for optimization

if command -v nvcc &> /dev/null; then
    CUDA_HOME=$(dirname $(dirname $(which nvcc)))
elif [ -d "/is/software/nvidia/cuda-11.8" ]; then
    CUDA_HOME=/is/software/nvidia/cuda-11.8
else
    echo "CUDA not found! Please install or add nvcc to PATH."
    return 1
fi

export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA detected at: $CUDA_HOME"
echo "LD_LIBRARY_PATH detected at: $LD_LIBRARY_PATH"

export GPU="0"
export CUDA_VISIBLE_DEVICES=$GPU

# Activate conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    PYTHON_ENV="$HOME/miniconda3/bin/activate"
    source "$PYTHON_ENV"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    PYTHON_ENV="$HOME/anaconda3/bin/activate"
    source "$PYTHON_ENV"
else
    echo "Conda not found in standard locations!"
fi

conda activate im2haircut

PORT=$((1000 + RANDOM % 5001))

echo "Starting multi-view training ..."
echo ""

export PYTHONPATH=./submodules/external/VOODOO3D-official:$PYTHONPATH
export PYTHONPATH=./submodules/external/GaussianHaircut:$PYTHONPATH

conf_path="static.conf"
exp_name="multiview_try"
data_path="./data"
folder_name="multiview_data"
RES="256"
NSTEPS="20"

# Get list of subjects (directories in multiview_data/)
SUBJECTS=$(ls -d $data_path/$folder_name/*/ 2>/dev/null | xargs -n 1 basename)

if [ -z "$SUBJECTS" ]; then
    echo "No subjects found in $data_path/$folder_name"
    exit 1
fi

echo "Found subjects: $SUBJECTS"
echo ""

for subject in $SUBJECTS; do
    echo "=========================================="
    echo "Processing subject: $subject"
    echo "=========================================="
    
    # Check if preprocessed data exists
    if [ ! -d "$data_path/$folder_name/$subject/resized_img_aligned" ]; then
        echo "Warning: No preprocessed data found for $subject, skipping..."
        continue
    fi
    
    # Count available views
    NUM_VIEWS=$(ls "$data_path/$folder_name/$subject/resized_img_aligned"/*.png 2>/dev/null | wc -l)
    echo "Using $NUM_VIEWS views for optimization"
    
    python run_image_reconstruction_mv.py \
        --conf_path ./configs/$conf_path \
        --savedir ./exps_inverse_stage/$exp_name/$folder_name/$subject \
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
        --subject "$subject" \
        --upsample_hairstyle True \
        --upsample_resolution $RES \
        --num_steps_coarse $NSTEPS \
        --folder_name $folder_name \
        --multiview True
    
    echo "Finished processing $subject"
    echo ""
done

echo "=========================================="
echo "Multi-view optimization pipeline finished!"
echo "=========================================="
