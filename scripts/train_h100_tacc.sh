#!/bin/bash
#===========================================================================
# SLURM Job Script — TACC H100: TRAINING (Single-Node, 2 GPUs)
# Rotation-Equivariant Rectified Flow
#===========================================================================
#
# 5000 training samples  ×  1 node  ×  2 H100 GPUs (80 GB each)
# Effective batch = 16 × 2 = 32  →  ~156 steps/epoch  →  ~234k updates
#
# Trains with FPS subsampling: 15k-point clouds → 2048 points.
# KNN-local attention (k=32) keeps memory at O(N·K).
# 80 GB VRAM allows batch=16 and train_n_points=2048 without grad checkpointing.
#
# Submit with:  sbatch scripts/train_h100_tacc.sh
#===========================================================================

#SBATCH -J rectflow-train-h100
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1                         # Single node
#SBATCH -n 1                         # 1 task (torchrun spawns 2 GPU workers)
#SBATCH -t 48:00:00
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

# Change to repo directory
SLURM_SUBMIT_DIR="/work/10692/ayuj/point_cloud_diffusion_model"
cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo " TRAINING — Rectified Flow Point Cloud"
echo "=========================================="
echo " Job ID:       $SLURM_JOB_ID"
echo " Job Name:     $SLURM_JOB_NAME"
echo " Node:         $SLURM_JOB_NODELIST"
echo " GPUs:         2x H100 (80 GB)"
echo "=========================================="

# ---- Single-node: no multi-node setup needed ----
NGPUS_PER_NODE=2

# ---- Hyperparameters ----
N_POINTS=15000          # Full resolution loaded from disk
TRAIN_N_POINTS=2048     # FPS subsample during training (15k → 2048); fits in 80 GB
KNN_K=32                # KNN neighbours for local attention (0 = global)
CHANNELS=128            # VN channel width
N_HEADS=8               # Attention heads
ENC_DEPTH=6             # Encoder transformer blocks
DEC_DEPTH=6             # Decoder transformer blocks
LATENT_DIM=256          # Shape latent z dimension
BATCH_SIZE=16           # Per-GPU batch size (effective = 16 × 2 = 32); backed down from 32 to avoid OOM
LR=1e-4                 # Linear scaling: base 1e-4 × (32 effective / 32 base) = 1e-4
EPOCHS=1500             # ~37-40h on 2 H100s (~90s/epoch), fits in 48h
WARMUP=15               # Warmup epochs (ramp LR from 0 → 2e-4 over 15 epochs)
GRAD_CLIP=1.0           # Max gradient norm
SAVE_EVERY=100           # Checkpoint every N epochs
VAL_EVERY=50           # Validation every N epochs
WANDB_PROJECT="dense-3d-point-correspondences"
WANDB_ENTITY="dense-3d-point-correspondences"

echo ""
echo " N_POINTS:       $N_POINTS (loaded from disk)"
echo " TRAIN_N_POINTS: $TRAIN_N_POINTS (FPS subsample for training)"
echo " KNN_K:          $KNN_K"
echo " BATCH_SIZE:     $BATCH_SIZE (per GPU, effective = $(( BATCH_SIZE * NGPUS_PER_NODE )))"
echo " LR:             $LR (linear scaled for effective batch $(( BATCH_SIZE * NGPUS_PER_NODE )))"
echo " EPOCHS:         $EPOCHS (~$(( EPOCHS * 4000 / (BATCH_SIZE * NGPUS_PER_NODE) )) gradient updates)"
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
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints/${SLURM_JOB_ID}"
LOGS_DIR="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_ID}"
mkdir -p "${CKPT_DIR}"
mkdir -p "${LOGS_DIR}"

# ------------------------------
# Activate conda environment
# ------------------------------
source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
conda activate rectflow
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Launch training (single-node, torchrun handles 2 GPUs) ----
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
        --ckpt_dir ${CKPT_DIR} \
        --save_every ${SAVE_EVERY} \
        --val_every ${VAL_EVERY} \
        --amp

echo "Training job $SLURM_JOB_ID finished."
