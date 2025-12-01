#!/bin/bash
#SBATCH --job-name="UniVAD_MaskGen"
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=education-ae-msc-ae
#SBATCH --output=/scratch/djjagt/univad/logs/masks_%j.out
#SBATCH --error=/scratch/djjagt/univad/logs/masks_%j.err

# ========================================================================
# UniVAD - Mask Generation (SAM + GroundingDINO)
# ========================================================================
# Runs segment_single_class.py on GPU to generate CFA masks
# Expected dataset location:
#   /scratch/<user>/datasets/<dataset>/
#
# Masks will be saved to:
#   /scratch/<user>/univad/masks/<dataset>/<class>/
#
# SAM checkpoints must be placed here:
#   /scratch/<user>/univad/pretrained_ckpts/
# ========================================================================

# ------------------------------------------
# Base Directories on DelftBlue
# ------------------------------------------

USER_SCRATCH="/scratch/${USER}"
UNIVAD_ROOT="${USER_SCRATCH}/univad"

DATA_PATH="${USER_SCRATCH}/datasets"               # where mvtec / sick / cognex are stored
MASKS_PATH="${UNIVAD_ROOT}/masks"                  # output masks
CKPTS_DIR="${UNIVAD_ROOT}/pretrained_ckpts"        # SAM checkpoints
LOG_DIR="${UNIVAD_ROOT}/logs"
RUNS_DIR="${UNIVAD_ROOT}/runs"

mkdir -p "$MASKS_PATH" "$CKPTS_DIR" "$LOG_DIR" "$RUNS_DIR"

# ------------------------------------------
# Create per-job run directory
# ------------------------------------------

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
jobid=${SLURM_JOB_ID:-manual}
RUN_DIR="${RUNS_DIR}/${jobid}_${timestamp}"
mkdir -p "$RUN_DIR"

export RUN_DIR
export GPU_LOG_PATH="${RUN_DIR}/gpu_status.txt"

echo "================================================="
echo " UniVAD Mask Generator"
echo " Job ID: $jobid"
echo " Run folder: $RUN_DIR"
echo " Dataset root: $DATA_PATH"
echo " Mask output: $MASKS_PATH"
echo " Checkpoints:  $CKPTS_DIR"
echo "================================================="

# ========================================================================
# Load Modules & Conda Environment
# ========================================================================
module purge
module load slurm
module load 2024r1
module load miniconda3
module load cuda/12.1

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/.conda/envs/univad

# ========================================================================
# CRITICAL LIBRARY FIXES (For GroundingDINO C++ Ops)
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
# CONFIGURATION (YOU EDIT THESE)
# ========================================================================

# Which dataset to process?
DATASET="sick"   # mvtec | sick | cognex | visa | mvtec_loco | oct17 | medical

# Class inside dataset
CLASS_NAME="None"    # "None" = all classes

# Image limit (-1 for all images)
NUM_IMAGES=-1

# Which SAM model to use?
SAM_VARIANT="vit_h"   # vit_b | vit_h
SAM_CHECKPOINT="${CKPTS_DIR}/sam_hq_vit_h.pth"

# Dataset root on scratch
DATA_ROOT="${DATA_PATH}/${DATASET}"

# ========================================================================
# Build Python Command
# ========================================================================

CMD="python segment_single_class.py \
    --dataset $DATASET \
    --mask_path $MASKS_PATH \
    --dataset_root $DATA_ROOT \
    --num_images $NUM_IMAGES \
    --sam_variant $SAM_VARIANT \
    --sam_checkpoint $SAM_CHECKPOINT \
    --phases train test"

if [ "$CLASS_NAME" != "None" ]; then
    CMD="${CMD} --class_name $CLASS_NAME"
fi

# ========================================================================
# RUN JOB
# ========================================================================

echo "================================================="
echo " RUNNING COMMAND:"
echo " $CMD"
echo "================================================="

srun --unbuffered $CMD
EXIT_CODE=$?

# ========================================================================
# POST-RUN LOGGING
# ========================================================================

echo "GPU status after:" >> "$GPU_LOG_PATH"
nvidia-smi >> "$GPU_LOG_PATH"

echo "================================================="
echo " Mask Generation Finished"
echo " Exit code: $EXIT_CODE"
echo " Masks saved to: $MASKS_PATH"
echo " Logs saved to: $LOG_DIR"
echo " Run folder: $RUN_DIR"
echo "================================================="

conda deactivate
exit $EXIT_CODE
