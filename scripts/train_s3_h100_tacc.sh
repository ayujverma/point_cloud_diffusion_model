#!/bin/bash
#===========================================================================
# SLURM Job Script — Stampede3 H100: TRAINING (Single-Node, 4 GPUs)
# Rotation-Equivariant Rectified Flow
#===========================================================================
#
# 5000 training samples  ×  1 node  ×  4 H100 SXM5 GPUs (96 GB each)
# Effective batch = 16 × 4 = 64  →  ~79 steps/epoch  →  ~118k updates total
#
# Submit with:  sbatch scripts/train_s3_h100_tacc.sh
#===========================================================================

#SBATCH -J rectflow-train-s3
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p h100
#SBATCH -N 1                         # Single node
#SBATCH -n 1                         # 1 task (torchrun spawns 4 GPU workers)
#SBATCH -t 48:00:00                  # Max job time is 48 hours for the h100 queue
#SBATCH -A ASC26027
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
echo " TRAINING — Rectified Flow Point Cloud"
echo "=========================================="
echo " Job ID:       $SLURM_JOB_ID"
echo " Job Name:     $SLURM_JOB_NAME"
echo " Node:         $SLURM_JOB_NODELIST"
echo " GPUs:         4x H100 SXM5 (96 GB)"
echo "=========================================="

# ---- Stampede3 H100 nodes have 4 GPUs ----
NGPUS_PER_NODE=4

# ---- Hyperparameters ----
N_POINTS=15000          # Full resolution loaded from disk
TRAIN_N_POINTS=2048     # FPS subsample during training; fits comfortably in 96GB
KNN_K=32                # KNN neighbours for local attention (0 = global)
CHANNELS=128            # VN channel width
N_HEADS=8               # Attention heads
ENC_DEPTH=6             # Encoder transformer blocks
DEC_DEPTH=6             # Decoder transformer blocks
LATENT_DIM=256          # Shape latent z dimension
BATCH_SIZE=16           # Per-GPU batch size (effective = 16 × 4 = 64); backed down from 32 to avoid OOM
LR=2e-4                 # Linear scaling: base 1e-4 × (64 effective / 32 base) = 2e-4
EPOCHS=1500             # Fits smoothly within hours limit on 4 H100s
WARMUP=20               # Warmup epochs (scaled slightly up for larger batch)
GRAD_CLIP=1.0           # Max gradient norm
SAVE_EVERY=100          # Checkpoint every N epochs
VAL_EVERY=50            # Validation every N epochs
WANDB_PROJECT="dense-3d-point-correspondences"
WANDB_ENTITY="dense-3d-point-correspondences"

echo ""
echo " N_POINTS:       $N_POINTS (loaded from disk)"
echo " TRAIN_N_POINTS: $TRAIN_N_POINTS (FPS subsample for training)"
echo " KNN_K:          $KNN_K"
echo " BATCH_SIZE:     $BATCH_SIZE (per GPU, effective = $(( BATCH_SIZE * NGPUS_PER_NODE )))"
echo " LR:             $LR (linear scaled for effective batch $(( BATCH_SIZE * NGPUS_PER_NODE )))"
echo " EPOCHS:         $EPOCHS"
echo " WARMUP:         $WARMUP epochs"
echo "=========================================="

# ---- Stage data to local scratch for fast I/O ----
# Note: Stampede3 H100 has 3.5TB /tmp partition
LOCAL_SCRATCH="/tmp/${USER}/human_pc15k"
DATA_ROOT="${REPO_DIR}/data/human"

echo "Staging data to $LOCAL_SCRATCH ..."
if [ ! -d "$LOCAL_SCRATCH/train" ]; then
    mkdir -p "$LOCAL_SCRATCH"
    cp -r "$DATA_ROOT/train" "$LOCAL_SCRATCH/"
    cp -r "$DATA_ROOT/val"   "$LOCAL_SCRATCH/"
    cp -r "$DATA_ROOT/test"  "$LOCAL_SCRATCH/"
fi
echo "Data staging complete."

# ---- Create directories ----
CKPT_DIR="${REPO_DIR}/checkpoints/${SLURM_JOB_ID}"
LOGS_DIR="${REPO_DIR}/logs/${SLURM_JOB_ID}"
mkdir -p "${CKPT_DIR}"
mkdir -p "${LOGS_DIR}"

# ------------------------------
# Activate conda environment
# ------------------------------
source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
conda activate rectflow
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Launch training (single-node, torchrun handles 4 GPUs) ----
echo ""
echo "Launching torchrun with ${NGPUS_PER_NODE} GPUs..."

torchrun \
    --nproc_per_node=${NGPUS_PER_NODE} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    ${REPO_DIR}/train.py \
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
