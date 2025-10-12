# Prerequisites:

# 1. Install CUDA 11.8
#    Follow intructions on https://developer.nvidia.com/cuda-11-8-0-download-archive
#    Make sure that
#      -   PATH includes <CUDA_DIR>/bin
#      -   LD_LIBRARY_PATH includes <CUDA_DIR>/lib64
#    If needed, restart bash environment

#    The environment was tested only with this CUDA version

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin


export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA detected at: $CUDA_HOME"
echo "LD_LIBRARY_PATH detected at: $LD_LIBRARY_PATH"


# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"

conda activate clip

# Save parent dir
# PROJECT_DIR=$PWD
GAUSSIAN_HAIRCUT_PATH="submodules/external/GaussianHaircut/ext"
cd $GAUSSIAN_HAIRCUT_PATH

bash install.sh
cd $PROJECT_DIR

# Download Im2Haircut files:
gdown https://drive.google.com/uc?id=1788vcfmdXIJKePOmBYVjC1Ts8IgeZb-_
tar -xzvf data.tar.gz

# Download Im2Haircut checkpoints:
gdown https://drive.google.com/uc?id=1uOuJx8kO22IZS3WTOeA5IQMw4cHXyamg
tar -xzvf pretrained_models.tar.gz