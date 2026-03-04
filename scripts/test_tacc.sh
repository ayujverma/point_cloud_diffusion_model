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

#SBATCH --job-name=rectflow-test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --export=ALL

# ---- Module setup ----
module purge
module load gcc/11.2.0
module load cuda/12.0
module load python3/3.10
module load tacc-apptainer 2>/dev/null || true

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

# ---- Activate environment ----
if [ -d "${SLURM_SUBMIT_DIR}/venv" ]; then
    source "${SLURM_SUBMIT_DIR}/venv/bin/activate"
elif [ -d "$HOME/miniconda3" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate rectflow 2>/dev/null || true
fi

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
