#!/bin/bash
# Preprocessing script for multi-view data
# Data structure expected:
# multiview_data/
# ├── subject_name/
# │   └── img/
# │       ├── view_000.png  (frontal view)
# │       ├── view_001.png
# │       └── ...

source ~/.bashrc
echo "CUDA detected at: $CUDA_HOME"
echo "LD_LIBRARY_PATH detected at: $LD_LIBRARY_PATH"

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda not found in standard locations!"
fi

path="./data"
path_set="multiview_data"

PROJECT_DIR=$(pwd)
SAM_PATH="./checkpoints/img2hairstep/SAM-models/sam_vit_h_4b8939.pth"

# Get list of subjects
SUBJECTS=$(ls -d $path/$path_set/*/ 2>/dev/null | xargs -n 1 basename)

if [ -z "$SUBJECTS" ]; then
    echo "No subjects found in $path/$path_set"
    exit 1
fi

echo "Found subjects: $SUBJECTS"
echo ""

# Process each subject
for subject in $SUBJECTS; do
    echo "=========================================="
    echo "Processing subject: $subject"
    echo "=========================================="
    
    SUBJECT_PATH="$path/$path_set/$subject"
    
    # Check if img folder exists
    if [ ! -d "$SUBJECT_PATH/img" ]; then
        echo "Warning: No img folder found for $subject, skipping..."
        continue
    fi
    
    # Count views
    NUM_VIEWS=$(ls "$SUBJECT_PATH/img"/*.png 2>/dev/null | wc -l)
    echo "Found $NUM_VIEWS views for $subject"
    
    # Step 1: Hair processing (HairStep - segmentation and strand maps)
    echo "Step 1: Running HairStep for segmentation and strand maps..."
    conda deactivate && conda activate clip
    
    cd ./submodules/external/HairStep
    python scripts/img2masks.py --root_real_imgs "$PROJECT_DIR/$SUBJECT_PATH" --checkpoint_sam $SAM_PATH
    python scripts/img2strand.py --root_real_imgs "$PROJECT_DIR/$SUBJECT_PATH"
    cd $PROJECT_DIR
    
    echo "Finished direction maps and silhouette estimation for $subject."
    
    # Step 2: Orientation maps and confidence maps (Gabor filter)
    echo "Step 2: Computing orientation maps..."
    python ./preprocess_dataset/calc_gabor_mask.py \
        --img_path "$SUBJECT_PATH/resized_img" \
        --path_to_save "$SUBJECT_PATH/orientation_maps/" \
        --path_to_save_conf "$SUBJECT_PATH/confidence_maps"
    
    echo "Finished gabor maps estimation for $subject."
    
    # Step 3: Depth processing (Apple DepthPro)
    echo "Step 3: Computing depth maps..."
    conda deactivate && conda activate im2haircut
    
    cd ./submodules/external/ml-depth-pro
    depth-pro-run -i "$PROJECT_DIR/$SUBJECT_PATH/resized_img" -o "$PROJECT_DIR/$SUBJECT_PATH/depth_apple_pro"
    cd $PROJECT_DIR
    
    echo "Finished depth estimation for $subject."
    
    # Step 4: Data alignment
    echo "Step 4: Aligning data..."
    python ./preprocess_dataset/calc_alignment.py \
        --img_path "$SUBJECT_PATH/resized_img" \
        --hair_path "$SUBJECT_PATH/seg" \
        --all_paths_for_processing seg body_img orientation_maps depth_apple_pro strand_map \
        --gt_img_path "$PROJECT_DIR/data/aligned_image.png"
    
    echo "Finished data aligning for $subject."
    
    # Step 5: Projection matrix calculation
    echo "Step 5: Computing projection matrices..."
    conda deactivate && conda activate clip
    
    # Non-aligned version
    python ./preprocess_dataset/calc_proj_matx.py --root_path "$SUBJECT_PATH"
    
    # Aligned version
    python ./preprocess_dataset/calc_proj_matx.py --root_path "$SUBJECT_PATH" --save_postfix "_aligned"
    
    echo "Finished projection matrix calculation for $subject."
    echo ""
done

echo "=========================================="
echo "Multi-view preprocessing pipeline finished!"
echo "=========================================="
