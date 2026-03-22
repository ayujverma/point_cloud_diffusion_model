#!/bin/bash
#===========================================================================
# SLURM Job Script — Stampede3 H100: TRAINING (Single-Node, 4 GPUs)
# Rotation-Equivariant Rectified Flow
#===========================================================================
#
# 5000 training samples  ×  1 node  ×  4 H100 SXM5 GPUs (96 GB each)
# Effective batch = 16 × 4 = 64  →  ~79 steps/epoch  →  ~118k updates total
#
# Loads full 15k-point clouds; FPS subsamples to 2048 in dataset.
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
N_POINTS=2048           # Template size and inference resolution
TRAIN_N_POINTS=2048     # FPS subsample size (dataset loads full 15k, FPS in __getitem__)
KNN_K=32                # KNN neighbours for local attention (0 = global)
CHANNELS=128            # VN channel width
N_HEADS=8               # Attention heads
ENC_DEPTH=6             # Encoder transformer blocks
DEC_DEPTH=6             # Decoder transformer blocks
LATENT_DIM=256          # Shape latent z dimension
BATCH_SIZE=16           # Per-GPU batch size (effective = 16 × 4 = 64)
LR=2e-4                 # Linear scaling: base 1e-4 × (64 effective / 32 base) = 2e-4
EPOCHS=1500             # Fits smoothly within hours limit on 4 H100s
WARMUP=20               # Warmup epochs
GRAD_CLIP=1.0           # Max gradient norm
SAVE_EVERY=100          # Checkpoint every N epochs
VAL_EVERY=50            # Validation every N epochs
VIS_EVERY=50            # Visualization every N epochs
WANDB_PROJECT="dense-3d-point-correspondences"
WANDB_ENTITY="dense-3d-point-correspondences"

# ---- Surface / OT loss hyperparameters ----
LAMBDA_OT=0.1
LAMBDA_REG=0.001
LAMBDA_CHAMFER=0.1
LAMBDA_REPULSION=0.01
SINKHORN_ITERS=50
SINKHORN_REG=0.01
TEMPLATE_REG_RADIUS=1.5
LAMBDA_REG_DECAY=0.995

echo ""
echo " N_POINTS:       $N_POINTS (template size)"
echo " TRAIN_N_POINTS: $TRAIN_N_POINTS (FPS subsample from 15k)"
echo " KNN_K:          $KNN_K"
echo " BATCH_SIZE:     $BATCH_SIZE (per GPU, effective = $(( BATCH_SIZE * NGPUS_PER_NODE )))"
echo " LR:             $LR"
echo " EPOCHS:         $EPOCHS"
echo " VIS_EVERY:      $VIS_EVERY epochs"
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
VIS_DIR="${CKPT_DIR}/vis"
LOGS_DIR="${REPO_DIR}/logs/${SLURM_JOB_ID}"
mkdir -p "${CKPT_DIR}"
mkdir -p "${VIS_DIR}"
mkdir -p "${LOGS_DIR}"

# ---- Compute mean shape for template initialization ----
# This gives the Sinkhorn alignment a massive head start vs. a Fibonacci sphere.
MEAN_SHAPE="${REPO_DIR}/data/mean_shape_${N_POINTS}.npy"

if [ ! -f "$MEAN_SHAPE" ]; then
    echo ""
    echo "=========================================="
    echo " Computing mean training shape..."
    echo " (This runs once, ~10-30 minutes)"
    echo "=========================================="

    # Activate conda first for this step
    source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
    conda activate rectflow

    python ${REPO_DIR}/compute_mean_shape.py \
        --data_root ${LOCAL_SCRATCH} \
        --n_points ${N_POINTS} \
        --output ${MEAN_SHAPE}
    echo "Mean shape computed: $MEAN_SHAPE"
else
    echo "Mean shape already exists: $MEAN_SHAPE"
    # Still need to activate conda
    source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
    conda activate rectflow
fi

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
        --template_init ${MEAN_SHAPE} \
        --lambda_ot ${LAMBDA_OT} \
        --lambda_reg ${LAMBDA_REG} \
        --lambda_chamfer ${LAMBDA_CHAMFER} \
        --lambda_repulsion ${LAMBDA_REPULSION} \
        --sinkhorn_iters ${SINKHORN_ITERS} \
        --sinkhorn_reg ${SINKHORN_REG} \
        --template_reg_radius ${TEMPLATE_REG_RADIUS} \
        --lambda_reg_decay ${LAMBDA_REG_DECAY} \
        --vis_every ${VIS_EVERY} \
        --vis_dir ${VIS_DIR} \
        --use_hard_assignment \
        --amp

echo "Training job $SLURM_JOB_ID finished."
