#!/bin/bash
#SBATCH --job-name="UniVAD_Test"
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M     
#SBATCH --account=education-ae-msc-ae
#SBATCH --output=/scratch/djjagt/univad/logs/univad_%j_test.out
#SBATCH --error=/scratch/djjagt/univad/logs/univad_%j_test.err

# ========================================================================
# UniVAD - Test / Inference Script
# ========================================================================
# Runs test_univad_no_pixel.py on GPU
# Expects:
#   /scratch/$USER/datasets/sick/...
#   /scratch/$USER/univad/masks/...
# ========================================================================

# ------------------------------------------
# Base Directories on DelftBlue
# ------------------------------------------
USER_SCRATCH="/scratch/${USER}"
UNIVAD_ROOT="${USER_SCRATCH}/univad"

DATA_PATH="${USER_SCRATCH}/datasets"   # Input images
MASKS_PATH="${UNIVAD_ROOT}/masks"      # Input masks (from previous step)
RESULTS_PATH="${UNIVAD_ROOT}/results"  # Output folder
LOG_DIR="${UNIVAD_ROOT}/logs"
RUNS_DIR="${UNIVAD_ROOT}/runs"

mkdir -p "$DATA_PATH" "$MASKS_PATH" "$RESULTS_PATH" "$LOG_DIR" "$RUNS_DIR"

# ------------------------------------------
# Create per-job run directory
# ------------------------------------------
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
jobid=${SLURM_JOB_ID:-manual}
RUN_DIR="${RUNS_DIR}/${jobid}_test_${timestamp}"
mkdir -p "$RUN_DIR"

export RUN_DIR
export GPU_LOG_PATH="${RUN_DIR}/gpu_status.txt"

echo "================================================="
echo " UniVAD Test Job"
echo " Job ID: $jobid"
echo " Run folder: $RUN_DIR"
echo "================================================="

# ========================================================================
# Load Modules & Conda Environment (MATCHING COMPILATION STACK)
# ========================================================================
module purge
module load slurm
module load 2024r1         # Vital: Matches the stack we compiled GroundingDINO with
module load miniconda3
module load cuda/12.1      # Vital: Matches Pytorch CUDA version

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/.conda/envs/univad

# ========================================================================
# CRITICAL LIBRARY FIXES
# ========================================================================
TORCH_LIB_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib"
export LD_LIBRARY_PATH="${TORCH_LIB_PATH}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

echo "🔧 Library Path Configured:"
echo "   TORCH_LIB: $TORCH_LIB_PATH"

# ========================================================================
# OFFLINE HUGGINGFACE CONFIG
# ========================================================================
export HF_HOME="/scratch/${USER}/.cache/huggingface"
export HF_HUB_CACHE="/scratch/${USER}/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/${USER}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# CLIP / general cache (matches download_clip.py)
export XDG_CACHE_HOME="/scratch/${USER}/.cache"
export TORCH_HOME="/scratch/${USER}/.cache/torch"

mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME"

# ========================================================================
# GPU Verification
# ========================================================================
echo "🔍 Checking GPU availability..."
echo "GPUs visible: $(nvidia-smi --list-gpus)"

echo "GPU status before:" > "$GPU_LOG_PATH"
nvidia-smi >> "$GPU_LOG_PATH"

python - << 'EOF'
import torch
print("🔥 PyTorch Version:", torch.__version__)
print("🔥 CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🔥 GPU:", torch.cuda.get_device_name(0))
    print("🔥 CUDA:", torch.version.cuda)
EOF

# ========================================================================
# CONFIGURATION 
# ========================================================================
DATASET="sick"                  # mvtec | sick | cognex | visa
CLASS_NAME="None"               # or specific product ID (e.g. "90006036")
IMAGE_SIZE=336
K_SHOT=10
NUM_IMAGES=-1
DEVICE="cuda"

# Defaults: ViT-L-14-336 (CLIP) and dinov2_vitg14 (Giant DINO)
CLIP_MODEL=""       # Options: ViT-L-14-336, ViT-L-14, ViT-B-16
DINO_MODEL=""      # Options: dinov2_vitg14, dinov2_vitl14, dinov2_vitb14

USE_LIGHT=false
FORCE_TEXTURE=false
DISABLE_CFA=false
SAVE_ALL=true        # true = save all examples
SAVE_MAPS=true       # true = save anomaly maps

# ========================================================================
# Build Python command
# ========================================================================
CMD="python test_univad_no_pixel.py \
    --dataset $DATASET \
    --data_path $DATA_PATH/$DATASET \
    --masks_path $MASKS_PATH \
    --save_path $RESULTS_PATH \
    --image_size $IMAGE_SIZE \
    --k_shot $K_SHOT \
    --device $DEVICE"

if [ "$CLASS_NAME" != "None" ]; then
    CMD="$CMD --class_name $CLASS_NAME"
fi

if [ "$NUM_IMAGES" != "-1" ]; then
    CMD="$CMD --num_images $NUM_IMAGES"
fi

# Only add CLIP model if user provided one
if [ -n "$CLIP_MODEL" ]; then
    CMD="$CMD --clip_model $CLIP_MODEL"
fi

# Only add DINO model if user provided one
if [ -n "$DINO_MODEL" ]; then 
    CMD="$CMD --dino_model $DINO_MODEL"
fi


if [ "$USE_LIGHT" = true ]; then CMD="$CMD --light"; fi
if [ "$FORCE_TEXTURE" = true ]; then CMD="$CMD --force_texture"; fi
if [ "$DISABLE_CFA" = true ]; then CMD="$CMD --disable_cfa"; fi
if [ "$SAVE_ALL" = true ]; then CMD="$CMD --save_all_examples"; fi
if [ "$SAVE_MAPS" = true ]; then CMD="$CMD --save_anomaly_maps"; fi

# ========================================================================
# RUN JOB
# ========================================================================
echo "==================== COMMAND ====================="
echo "$CMD"
echo "==================================================="

srun --unbuffered $CMD
EXIT_CODE=$?

# ========================================================================
# POST-RUN LOGGING
# ========================================================================
echo "GPU status after:" >> "$GPU_LOG_PATH"
nvidia-smi >> "$GPU_LOG_PATH"

echo "================================================="
echo " UniVAD Job Finished (exit code: $EXIT_CODE)"
echo " End: $(date)"
echo "================================================="

conda deactivate
exit $EXIT_CODE
