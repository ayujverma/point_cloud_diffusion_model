#!/bin/bash
#===========================================================================
# SLURM Job Script — TACC A100: TRAINING (Single-Node, 3 GPUs)
# Rotation-Equivariant Rectified Flow
#===========================================================================
#
# 5000 training samples  ×  1 node  ×  3 A100 GPUs
# Effective batch = 32 × 3 = 96  →  ~52 steps/epoch  →  ~78k updates
#
# Trains with FPS subsampling: 15k-point clouds → 2048 points.
# KNN-local attention (k=32) keeps memory at O(N·K).
#
# Submit with:  sbatch scripts/train_tacc.sh
#===========================================================================

#SBATCH -J rectflow-train
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1                         # Single node (3 GPUs is enough for 5k samples)
#SBATCH -n 1                         # 1 task (torchrun spawns 3 GPU workers)
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=0                      # Use all available memory
#SBATCH -t 48:00:00
#SBATCH -A CCR25016
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ayuj@utexas.edu
#SBATCH --export=ALL

# ------------------------------
# Load system modules
# ------------------------------
module purge
module load cuda/12.2

echo "=========================================="
echo " TRAINING — Rectified Flow Point Cloud"
echo "=========================================="
echo " Job ID:       $SLURM_JOB_ID"
echo " Job Name:     $SLURM_JOB_NAME"
echo " Node:         $SLURM_JOB_NODELIST"
echo " GPUs:         3x A100"
echo "=========================================="

# ---- Single-node: no multi-node setup needed ----
NGPUS_PER_NODE=3

# ---- Hyperparameters ----
N_POINTS=15000          # Full resolution loaded from disk
TRAIN_N_POINTS=2048     # FPS subsample during training (15k → 2048)
KNN_K=32                # KNN neighbours for local attention (0 = global)
CHANNELS=128            # VN channel width
N_HEADS=8               # Attention heads
ENC_DEPTH=6             # Encoder transformer blocks
DEC_DEPTH=6             # Decoder transformer blocks
LATENT_DIM=256          # Shape latent z dimension
BATCH_SIZE=32           # Per-GPU batch size (effective = 32 × 3 = 96)
LR=3e-4                 # Linear scaling: base 1e-4 × 3 GPUs
EPOCHS=1500             # 1500 × 52 steps ≈ 78k gradient updates
WARMUP=15               # Longer warmup for larger effective LR
GRAD_CLIP=1.0           # Max gradient norm (stabilises training)
SAVE_EVERY=100           # Checkpoint every N epochs
VAL_EVERY=50             # Validation every N epochs
WANDB_PROJECT="Dense 3D Point Correspondences"
WANDB_ENTITY="ayuj-the-university-of-texas-at-austin"

echo ""
echo " N_POINTS:       $N_POINTS (loaded from disk)"
echo " TRAIN_N_POINTS: $TRAIN_N_POINTS (FPS subsample for training)"
echo " KNN_K:          $KNN_K"
echo " BATCH_SIZE:     $BATCH_SIZE (per GPU, effective = $(( BATCH_SIZE * NGPUS_PER_NODE )))"
echo " LR:             $LR (linear scaled for $NGPUS_PER_NODE GPUs)"
echo " EPOCHS:         $EPOCHS (~$(( EPOCHS * 5000 / (BATCH_SIZE * NGPUS_PER_NODE) )) gradient updates)"
echo " WARMUP:         $WARMUP epochs"
echo "=========================================="

# ---- Stage data to local scratch for fast I/O ----
LOCAL_SCRATCH="/tmp/${USER}/human_pc15k"
DATA_ROOT="${SLURM_SUBMIT_DIR}/data/human"

echo "Staging data to $LOCAL_SCRATCH ..."
if [ ! -d "$LOCAL_SCRATCH/train" ]; then
    mkdir -p "$LOCAL_SCRATCH"
    cp -r "$DATA_ROOT/train" "$LOCAL_SCRATCH/"
    cp -r "$DATA_ROOT/val"   "$LOCAL_SCRATCH/"
    cp -r "$DATA_ROOT/test"  "$LOCAL_SCRATCH/"
fi
echo "Data staging complete."

# ---- Create directories ----
mkdir -p "${SLURM_SUBMIT_DIR}/logs"
mkdir -p "${SLURM_SUBMIT_DIR}/checkpoints"

# ------------------------------
# Activate conda environment
# ------------------------------
source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
conda activate rectflow
export PYTHONUNBUFFERED=1

# ---- Launch training (single-node, torchrun handles 3 GPUs) ----
echo ""
echo "Launching torchrun with ${NGPUS_PER_NODE} GPUs..."

torchrun \
    --nproc_per_node=${NGPUS_PER_NODE} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    ${SLURM_SUBMIT_DIR}/train.py \
        --data_root ${LOCAL_SCRATCH} \
        --n_points ${N_POINTS} \
        --train_n_points ${TRAIN_N_POINTS} \
        --knn_k ${KNN_K} \
        --channels ${CHANNELS} \
        --n_heads ${N_HEADS} \
        --enc_depth ${ENC_DEPTH} \
        --dec_depth ${DEC_DEPTH} \
        --latent_dim ${LATENT_DIM} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --epochs ${EPOCHS} \
        --warmup_epochs ${WARMUP} \
        --grad_clip ${GRAD_CLIP} \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_entity "${WANDB_ENTITY}" \
        --ckpt_dir ${SLURM_SUBMIT_DIR}/checkpoints \
        --save_every ${SAVE_EVERY} \
        --val_every ${VAL_EVERY} \
        --amp

echo "Training job $SLURM_JOB_ID finished."
