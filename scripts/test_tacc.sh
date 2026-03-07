#!/bin/bash
#===========================================================================
# SLURM Job Script — TACC A100: TESTING (Full-Resolution Inference)
# Rotation-Equivariant Rectified Flow
#===========================================================================
#
# Runs full-resolution inference: flows ALL 15k template points through
# the learned velocity field to each test target.  Single-node, single-GPU
# since inference is sequential per shape (no DDP needed).
#
# Submit with:  sbatch scripts/test_tacc.sh
#
# Dependencies:
#   - A trained checkpoint at $CHECKPOINT (edit below)
#===========================================================================

#SBATCH -J rectflow-test
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1                         # Single node (3 GPUs is enough for 5k samples)
#SBATCH -n 1                         # 1 task (torchrun spawns 3 GPU workers)                    # Use all available memory
#SBATCH -t 24:00:00
#SBATCH -A CCR25016
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ayuj@utexas.edu

# ------------------------------
# Load system modules
# ------------------------------
module purge
module load cuda/12.2
export CUDA_HOME=$TACC_CUDA_DIR
export CUDA_PATH=$TACC_CUDA_DIR

# Change to repo directory so Python imports resolve
cd ${SLURM_SUBMIT_DIR}

echo "=========================================="
echo " TESTING — Full-Resolution Inference"
echo "=========================================="
echo " Job ID:       $SLURM_JOB_ID"
echo " Node:         $SLURM_JOB_NODELIST"
echo " GPU:          1x A100"
echo "=========================================="

# ---- Inference settings (edit here) ----
CHECKPOINT="${SLURM_SUBMIT_DIR}/checkpoints/best.pt"
DATA_ROOT="${SLURM_SUBMIT_DIR}/data/human"
SPLIT="test"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/full_res"

# Model architecture (must match training)
N_POINTS=15000
CHANNELS=128
N_HEADS=8
ENC_DEPTH=6
DEC_DEPTH=6
LATENT_DIM=256
KNN_K=32

# ODE integration
N_STEPS=50
METHOD="midpoint"      # "euler" or "midpoint"
MAX_SAMPLES=-1          # -1 = all test shapes

echo ""
echo " Checkpoint:   $CHECKPOINT"
echo " Data root:    $DATA_ROOT"
echo " Split:        $SPLIT"
echo " Template pts: $N_POINTS (full resolution)"
echo " KNN-k:        $KNN_K"
echo " ODE steps:    $N_STEPS ($METHOD)"
echo " Output:       $OUTPUT_DIR"
echo "=========================================="

# ---- Verify checkpoint exists ----
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "       Train a model first with: sbatch scripts/train_tacc.sh"
    exit 1
fi

# ---- Stage data to local scratch ----
LOCAL_SCRATCH="/tmp/${USER}/human_pc15k"

echo "Staging data to $LOCAL_SCRATCH ..."
if [ ! -d "$LOCAL_SCRATCH/${SPLIT}" ]; then
    mkdir -p "$LOCAL_SCRATCH"
    cp -r "${DATA_ROOT}/${SPLIT}" "$LOCAL_SCRATCH/"
fi
echo "Data staging complete."

# ---- Create output directory ----
mkdir -p "$OUTPUT_DIR"
mkdir -p "${SLURM_SUBMIT_DIR}/logs"

# ------------------------------
# Activate conda environment
# ------------------------------
source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
conda activate rectflow
export PYTHONUNBUFFERED=1

# ---- Run full-resolution inference ----
echo ""
echo "Starting full-resolution inference..."
echo "  Flowing all ${N_POINTS} template points → each ${SPLIT} target"
echo ""

python ${SLURM_SUBMIT_DIR}/test.py \
    --checkpoint ${CHECKPOINT} \
    --data_root ${LOCAL_SCRATCH} \
    --split ${SPLIT} \
    --n_points ${N_POINTS} \
    --channels ${CHANNELS} \
    --n_heads ${N_HEADS} \
    --enc_depth ${ENC_DEPTH} \
    --dec_depth ${DEC_DEPTH} \
    --latent_dim ${LATENT_DIM} \
    --knn_k ${KNN_K} \
    --n_steps ${N_STEPS} \
    --method ${METHOD} \
    --max_samples ${MAX_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --device cuda

echo ""
echo "=========================================="
echo " Inference complete."
echo " Results saved to: $OUTPUT_DIR"
echo "=========================================="
echo "Job $SLURM_JOB_ID finished."
