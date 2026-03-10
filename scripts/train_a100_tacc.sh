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
#SBATCH -n 1                         # 1 task (torchrun spawns 3 GPU workers)                    # Use all available memory
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
REPO_DIR="/work/10692/ayuj/point_cloud_diffusion_model"
cd "${REPO_DIR}"

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
N_POINTS=1024           # Template + data resolution (matches inference)
TRAIN_N_POINTS=0        # 0 = no FPS subsampling (already at target resolution)
KNN_K=32                # KNN neighbours for local attention (0 = global)
CHANNELS=128            # VN channel width
N_HEADS=8               # Attention heads
ENC_DEPTH=6             # Encoder transformer blocks
DEC_DEPTH=6             # Decoder transformer blocks
LATENT_DIM=256          # Shape latent z dimension
BATCH_SIZE=16           # Per-GPU batch size (effective = 16 × 3 = 48); use 32 if no OOM
LR=1.5e-4               # Linear scaling: base 1e-4 × 3 GPUs × (16/32 batch) = 1.5e-4
EPOCHS=1500             # fits in 48h with 3 GPUs (~90-100s/epoch)
WARMUP=10               # Warmup epochs
GRAD_CLIP=1.0           # Max gradient norm (stabilises training)
SAVE_EVERY=100          # Checkpoint every N epochs
VAL_EVERY=50            # Validation + visualization every N epochs
WANDB_PROJECT="dense-3d-point-correspondences"
WANDB_ENTITY="dense-3d-point-correspondences"

# ---- Surface / OT loss hyperparameters ----
LAMBDA_OT=0.1
LAMBDA_REG=0.001
LAMBDA_CHAMFER=0.1
LAMBDA_REPULSION=0.01
SINKHORN_ITERS=100
SINKHORN_REG=0.01
TEMPLATE_REG_RADIUS=2.0
LAMBDA_REG_DECAY=0.995

echo ""
echo " N_POINTS:       $N_POINTS (template + data resolution)"
echo " TRAIN_N_POINTS: $TRAIN_N_POINTS (0 = disabled, already at target res)"
echo " KNN_K:          $KNN_K"
echo " BATCH_SIZE:     $BATCH_SIZE (per GPU, effective = $(( BATCH_SIZE * NGPUS_PER_NODE )))"
echo " LR:             $LR (linear scaled for $NGPUS_PER_NODE GPUs)"
echo " EPOCHS:         $EPOCHS (~$(( EPOCHS * 5000 / (BATCH_SIZE * NGPUS_PER_NODE) )) gradient updates)"
echo " WARMUP:         $WARMUP epochs"
echo "=========================================="

# ---- Stage data to local scratch for fast I/O ----
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

# ---- Compute mean shape for template initialization ----
MEAN_SHAPE="${REPO_DIR}/data/mean_shape_2048.npy"

if [ ! -f "$MEAN_SHAPE" ]; then
    echo ""
    echo "Computing mean training shape..."
    source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
    conda activate rectflow
    python ${REPO_DIR}/compute_mean_shape.py \
        --data_root ${LOCAL_SCRATCH} \
        --n_points ${N_POINTS} \
        --output ${MEAN_SHAPE} \
        --hungarian_limit 4096
    echo "Mean shape computed: $MEAN_SHAPE"
else
    echo "Mean shape already exists: $MEAN_SHAPE"
    source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
    conda activate rectflow
fi

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
        --template_init ${MEAN_SHAPE} \
        --lambda_ot ${LAMBDA_OT} \
        --lambda_reg ${LAMBDA_REG} \
        --lambda_chamfer ${LAMBDA_CHAMFER} \
        --lambda_repulsion ${LAMBDA_REPULSION} \
        --sinkhorn_iters ${SINKHORN_ITERS} \
        --sinkhorn_reg ${SINKHORN_REG} \
        --template_reg_radius ${TEMPLATE_REG_RADIUS} \
        --lambda_reg_decay ${LAMBDA_REG_DECAY} \
        --use_hard_assignment \
        --amp

echo "Training job $SLURM_JOB_ID finished."
