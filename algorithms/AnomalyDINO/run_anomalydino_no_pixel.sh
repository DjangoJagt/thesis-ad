#!/bin/bash
#SBATCH --job-name="AnomalyDINO_Test"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-ae-msc-ae
#SBATCH --output=/scratch/djjagt/anomalydino/logs/anomalydino_%j.out
#SBATCH --error=/scratch/djjagt/anomalydino/logs/anomalydino_%j.err

# ========================================================================
# AnomalyDINO - No Pixel Ground Truth Runner (DelftBlue)
# ========================================================================
# Runs run_anomalydino_no_pixel.py on GPU 
# ========================================================================

# ------------------------------------------
# Base Directories on DelftBlue
# ------------------------------------------
USER_SCRATCH="/scratch/${USER}"         
ANOMALYDINO_ROOT="${USER_SCRATCH}/anomalydino"  

DATA_PATH="${USER_SCRATCH}/datasets"              # Input datasets live here
RESULTS_PATH="${ANOMALYDINO_ROOT}/results" # Output directory (safe to change)
LOG_DIR="${ANOMALYDINO_ROOT}/logs"
RUNS_DIR="${ANOMALYDINO_ROOT}/runs"

mkdir -p "${DATA_PATH}" "${RESULTS_PATH}" "${LOG_DIR}" "${RUNS_DIR}"

# ------------------------------------------
# Create per-job run directory
# ------------------------------------------
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
jobid=${SLURM_JOB_ID:-manual}
RUN_DIR="${RUNS_DIR}/${jobid}_test_${timestamp}"
mkdir -p "${RUN_DIR}"

export RUN_DIR
export GPU_LOG_PATH="${RUN_DIR}/gpu_status.txt"

echo "================================================="
echo " AnomalyDINO Test Job"
echo " Job ID: ${jobid}"
echo " Run folder: ${RUN_DIR}"
echo "================================================="

# ------------------------------------------
# Load Modules & Conda Environment
# ------------------------------------------
module purge
module load slurm
module load 2024r1
module load miniconda3
module load cuda/12.1

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/.conda/envs/anomalydino

# ------------------------------------------
# Library Fixes
# ------------------------------------------
TORCH_LIB_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib"
export LD_LIBRARY_PATH="${TORCH_LIB_PATH}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

echo "🔧 Library Path Configured:"
echo "   TORCH_LIB: ${TORCH_LIB_PATH}"

# ------------------------------------------
# Optional Offline Cache Config (uncomment if needed)
# ------------------------------------------
export HF_HOME="${USER_SCRATCH}/.cache/huggingface"
# export HF_HUB_CACHE="${HF_HOME}"
# export TRANSFORMERS_CACHE="${HF_HOME}"
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export XDG_CACHE_HOME="${USER_SCRATCH}/.cache"
export TORCH_HOME="${USER_SCRATCH}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" #"${XDG_CACHE_HOME}" 

# ------------------------------------------
# GPU Verification
# ------------------------------------------
echo "🔍 Checking GPU availability..."
echo "GPUs visible: $(nvidia-smi --list-gpus)"

echo "GPU status before:" > "${GPU_LOG_PATH}"
nvidia-smi >> "${GPU_LOG_PATH}"

python - <<'EOF'
import torch
print("🔥 PyTorch Version:", torch.__version__)
print("🔥 CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🔥 GPU:", torch.cuda.get_device_name(0))
    print("🔥 CUDA:", torch.version.cuda)
EOF

# ------------------------------------------
# Experiment Configuration
# ------------------------------------------
DATASET="sick"
MODEL_NAME="dinov2_vits14"
RESOLUTION=448
PREPROCESS="agnostic_no_mask"
K_SHOT=10
SEED=0
DEVICE="cuda:0"
SAVE_EXAMPLES=true
OBJECTS=""                              # e.g. "90006036 90006124"

K_NEIGHBORS=1
KNN_METRIC="L2_normalized"
FAISS_ON_CPU=false
MASK_REF_IMAGES=false
TAG=""
                                    

# Ensure repo path exists and switch into AnomalyDINO folder

CMD="python run_anomalydino_no_pixel.py \
    --dataset ${DATASET} \
    --data_root $DATA_PATH/$DATASET \
    --save_path $RESULTS_PATH \
    --model_name ${MODEL_NAME} \
    --resolution ${RESOLUTION} \
    --preprocess ${PREPROCESS} \
    --shots ${K_SHOT} \
    --k_neighbors ${K_NEIGHBORS} \
    --knn_metric ${KNN_METRIC} \
    --seed ${SEED} \
    --device ${DEVICE}"

if [ "${FAISS_ON_CPU}" = true ]; then
    CMD="${CMD} --faiss_on_cpu"
fi
if [ "${MASK_REF_IMAGES}" = true ]; then
    CMD="${CMD} --mask_ref_images"
fi
if [ "${SAVE_EXAMPLES}" = true ]; then
    CMD="${CMD} --save_examples"
fi
if [ -n "${TAG}" ]; then
    CMD="${CMD} --tag ${TAG}"
fi
if [ -n "${OBJECTS}" ]; then
    CMD="${CMD} --objects ${OBJECTS}"
fi

echo "==================== COMMAND ====================="
echo "${CMD}"
echo "==================================================="

srun --unbuffered ${CMD}
EXIT_CODE=$?

# ------------------------------------------
# Post-run Logging
# ------------------------------------------
echo "GPU status after:" >> "${GPU_LOG_PATH}"
nvidia-smi >> "${GPU_LOG_PATH}"

echo "================================================="
echo " AnomalyDINO Job Finished (exit code: ${EXIT_CODE})"
echo " End: $(date)"
echo "================================================="

conda deactivate
exit $EXIT_CODE
