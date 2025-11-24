#!/bin/bash
#SBATCH --job-name="UniVAD_Test"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2          # max allowed on gpu-a100-small
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-AE-MSc-AE
#SBATCH --output=/scratch/%u/univad/logs/univad_%j_test.out
#SBATCH --error=/scratch/%u/univad/logs/univad_%j_test.err

# ========================================================================
# UniVAD - DelftBlue SLURM Script
# ========================================================================
# Uses conda env at:
#   /scratch/$USER/.conda/envs/univad
#
# Expects:
#   /scratch/$USER/datasets/sick/...
#   /scratch/$USER/datasets/mvtec/...
#   /scratch/$USER/univad/masks/...
#   /scratch/$USER/univad/results/...
# ========================================================================

# Base paths
USER_SCRATCH="/scratch/${USER}"
UNIVAD_ROOT="${USER_SCRATCH}/univad"

DATA_PATH="${USER_SCRATCH}/datasets"   # /scratch/$USER/datasets/{sick,mvtec,...}
MASKS_PATH="${UNIVAD_ROOT}/masks"      # /scratch/$USER/univad/masks
RESULTS_PATH="${UNIVAD_ROOT}/results"  # /scratch/$USER/univad/results
LOG_DIR="${UNIVAD_ROOT}/logs"
RUNS_DIR="${UNIVAD_ROOT}/runs"

mkdir -p "$DATA_PATH" "$MASKS_PATH" "$RESULTS_PATH" "$LOG_DIR" "$RUNS_DIR"

# Per-job run directory
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
jobid=${SLURM_JOB_ID:-manual}
RUN_DIR="${RUNS_DIR}/${jobid}_${timestamp}"
mkdir -p "$RUN_DIR"

export RUN_DIR="$RUN_DIR"
export GPU_LOG_PATH="$RUN_DIR/nvidia_log.txt"

echo "================================================="
echo " UniVAD Job Started"
echo " Job ID: $jobid"
echo " Run directory: $RUN_DIR"
echo "================================================="

# ========================================================================
# Load modules (no stack or CUDA modules to avoid conflicts)
# ========================================================================
module purge
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/.conda/envs/univad

# Go to repo root (important!)
cd /home/${USER}/thesis-ad

# ========================================================================
# Verify GPU access
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
    print("🔥 CUDA Version Used by Torch:", torch.version.cuda)
EOF

# ========================================================================
# CONFIGURATION – MODIFY AS NEEDED
# ========================================================================
DATASET="sick"                  # mvtec | sick | cognex | visa
CLASS_NAME="None"               # or specific product ID (e.g. "90006036")
K_SHOT=10
NUM_IMAGES=-1
DEVICE="cuda"
USE_LIGHT=false
FORCE_TEXTURE=false
DISABLE_CFA=false
SAVE_ALL=true        # true = save all examples
SAVE_MAPS=true       # true = save anomaly maps

# ========================================================================
# Build Python command
# ========================================================================
CMD="python algorithms/UniVAD/test_univad_no_pixel.py \
    --dataset $DATASET \
    --data_path $DATA_PATH/$DATASET \
    --masks_path $MASKS_PATH \
    --save_path $RESULTS_PATH/$DATASET \
    --k_shot $K_SHOT \
    --device $DEVICE"

if [ "$CLASS_NAME" != "None" ]; then
    CMD="$CMD --class_name $CLASS_NAME"
fi

if [ "$NUM_IMAGES" != "-1" ]; then
    CMD="$CMD --num_images $NUM_IMAGES"
fi

if [ "$USE_LIGHT" = true ]; then
    CMD="$CMD --light"
fi

if [ "$FORCE_TEXTURE" = true ]; then
    CMD="$CMD --force_texture"
fi

if [ "$DISABLE_CFA" = true ]; then
    CMD="$CMD --disable_cfa"
fi

if [ "$SAVE_ALL" = true ]; then
    CMD="$CMD --save_all_examples"
else
    CMD="$CMD --save_examples"
fi

if [ "$SAVE_MAPS" = true ]; then
    CMD="$CMD --save_anomaly_maps"
fi

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
