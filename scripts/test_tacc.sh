#!/bin/bash
#===========================================================================
# SLURM Job Script — TACC: TESTING (Full-Resolution Inference)
# Rotation-Equivariant Rectified Flow
#===========================================================================
#
# Runs single-GPU full-resolution inference, then visualisation & evaluation.
#
# Before submitting, edit the SBATCH lines below to match your queue:
#   - Stampede3 H100:  -p h100       -A ASC26027
#   - Lonestar6 H100:      -p gpu-h100   -A CCR25016
#   - Lonestar6 A100:  -p gpu-a100   -A CCR25016
#
# Submit with:  sbatch scripts/test_tacc.sh
#===========================================================================

#SBATCH -J rectflow-test
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p h100                      # ← EDIT: h100 | gpu-h100 | gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A ASC26027                  # ← EDIT: ASC26027 | CCR25016
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ayuj@utexas.edu

# ------------------------------
# Load system modules
# ------------------------------
module purge
module load cuda/12.2
export CUDA_HOME=$TACC_CUDA_DIR
export CUDA_PATH=$TACC_CUDA_DIR

# Change to repo directory
REPO_DIR="/work/10692/ayuj/point_cloud_diffusion_model"
cd "${REPO_DIR}"

echo "=========================================="
echo " TESTING — Full-Resolution Inference"
echo "=========================================="
echo " Job ID:       $SLURM_JOB_ID"
echo " Node:         $SLURM_JOB_NODELIST"
echo "=========================================="

# ---- Inference settings (edit here) ----
TRAIN_JOB_ID="2921884"

CHECKPOINT="${REPO_DIR}/checkpoints/${TRAIN_JOB_ID}/best.pt"
DATA_ROOT="${REPO_DIR}/data/human"
SPLIT="test"
OUTPUT_DIR="${REPO_DIR}/results/model_${TRAIN_JOB_ID}/test_${SLURM_JOB_ID}"

# Model architecture (must match training)
N_POINTS=2048
CHANNELS=128
N_HEADS=8
ENC_DEPTH=6
DEC_DEPTH=6
LATENT_DIM=256
KNN_K=32

# ODE integration
N_STEPS=50
METHOD="midpoint"
MAX_SAMPLES=-1         # -1 = all test shapes

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
mkdir -p "${REPO_DIR}/logs"

# ------------------------------
# Activate conda environment
# ------------------------------
source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
conda activate rectflow
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==========================================================================
# STEP 1: Full-resolution inference (single GPU)
# ==========================================================================
echo ""
echo "=========================================="
echo " Step 1/3: Full-Resolution Inference"
echo "=========================================="
echo "  Flowing all ${N_POINTS} template points → each ${SPLIT} target"

python ${REPO_DIR}/test.py \
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

# ==========================================================================
# STEP 2: Visualise correspondence (10 targets, GIFs + grid)
# ==========================================================================
echo ""
echo "=========================================="
echo " Step 2/3: Visualising Correspondence"
echo "=========================================="

python ${REPO_DIR}/visualize.py \
    --checkpoint ${CHECKPOINT} \
    --data_root ${LOCAL_SCRATCH} \
    --n_targets 10 \
    --n_steps 30 \
    --n_points ${N_POINTS} \
    --vis_n_points 2048 \
    --channels ${CHANNELS} \
    --n_heads ${N_HEADS} \
    --enc_depth ${ENC_DEPTH} \
    --dec_depth ${DEC_DEPTH} \
    --latent_dim ${LATENT_DIM} \
    --knn_k ${KNN_K} \
    --output_dir ${OUTPUT_DIR}/visualizations \
    --device cuda

# ==========================================================================
# STEP 3: Evaluate alignment metrics
# ==========================================================================
echo ""
echo "=========================================="
echo " Step 3/3: Evaluating Alignment Metrics"
echo "=========================================="

python ${REPO_DIR}/evaluate.py \
    --results_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR}/eval_metrics

echo ""
echo "=========================================="
echo " All steps complete."
echo " Results:  $OUTPUT_DIR"
echo " Visuals:  $OUTPUT_DIR/visualizations"
echo " Metrics:  $OUTPUT_DIR/eval_metrics"
echo "=========================================="
echo "Job $SLURM_JOB_ID finished."
