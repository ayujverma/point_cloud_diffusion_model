#!/bin/bash
#===========================================================================
# SLURM Job Script — TACC A100: TRAINING
# Rotation-Equivariant Rectified Flow (Multi-Node DDP)
#===========================================================================
#
# Trains with FPS subsampling: 15k-point clouds are downsampled to 2048
# during training for memory efficiency.  KNN-local attention (k=32)
# keeps per-layer cost at O(N·K) instead of O(N²).
#
# Submit with:  sbatch scripts/train_tacc.sh
#===========================================================================

#SBATCH --job-name=rectflow-train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=0                      # Use all available memory
#SBATCH --time=48:00:00
#SBATCH --export=ALL

# ---- Module setup (adjust for your TACC system) ----
module purge
module load gcc/11.2.0
module load cuda/12.0
module load python3/3.10
module load tacc-apptainer 2>/dev/null || true

echo "=========================================="
echo " TRAINING — Rectified Flow Point Cloud"
echo "=========================================="
echo " Job ID:       $SLURM_JOB_ID"
echo " Job Name:     $SLURM_JOB_NAME"
echo " Nodes:        $SLURM_JOB_NODELIST"
echo " Tasks/Node:   $SLURM_NTASKS_PER_NODE"
echo " GPUs/Node:    3"
echo " Total GPUs:   $(( SLURM_NNODES * 3 ))"
echo "=========================================="

# ---- Distributed training variables ----
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

NNODES=$SLURM_NNODES
NGPUS_PER_NODE=3
WORLD_SIZE=$(( NNODES * NGPUS_PER_NODE ))

echo " MASTER_ADDR:  $MASTER_ADDR"
echo " MASTER_PORT:  $MASTER_PORT"
echo " WORLD_SIZE:   $WORLD_SIZE"

# ---- Hyperparameters (edit here) ----
N_POINTS=15000          # Full resolution loaded from disk
TRAIN_N_POINTS=2048     # FPS subsample during training
KNN_K=32                # KNN neighbours for local attention (0 = global)
CHANNELS=128
N_HEADS=8
ENC_DEPTH=6
DEC_DEPTH=6
LATENT_DIM=256
BATCH_SIZE=32           # Per-GPU batch size
LR=1e-4
EPOCHS=300
WARMUP=10
GRAD_CLIP=1.0
WANDB_PROJECT="rectified-flow-pc"

echo ""
echo " N_POINTS:       $N_POINTS (loaded from disk)"
echo " TRAIN_N_POINTS: $TRAIN_N_POINTS (FPS subsample for training)"
echo " KNN_K:          $KNN_K"
echo " BATCH_SIZE:     $BATCH_SIZE (per GPU)"
echo " EPOCHS:         $EPOCHS"
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

# ---- Activate environment ----
if [ -d "${SLURM_SUBMIT_DIR}/venv" ]; then
    source "${SLURM_SUBMIT_DIR}/venv/bin/activate"
elif [ -d "$HOME/miniconda3" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate rectflow 2>/dev/null || true
fi

# ---- Launch training (one torchrun per node via srun) ----
srun --nodes=1 --ntasks=1 --exclusive bash -c "
    export MASTER_ADDR=${MASTER_ADDR}
    export MASTER_PORT=${MASTER_PORT}

    # Compute NODE_RANK inside srun
    HOSTS=(\$(scontrol show hostnames \"$SLURM_JOB_NODELIST\"))
    ME=\$(hostname -s)
    for i in \"\${!HOSTS[@]}\"; do
        if [ \"\${HOSTS[\$i]}\" == \"\$ME\" ]; then
            export NODE_RANK=\$i
            break
        fi
    done

    echo \"Launching torchrun on \$ME (NODE_RANK=\$NODE_RANK)\"

    torchrun \\
        --nproc_per_node=${NGPUS_PER_NODE} \\
        --nnodes=${NNODES} \\
        --node_rank=\${NODE_RANK} \\
        --master_addr=${MASTER_ADDR} \\
        --master_port=${MASTER_PORT} \\
        ${SLURM_SUBMIT_DIR}/train.py \\
            --data_root ${LOCAL_SCRATCH} \\
            --n_points ${N_POINTS} \\
            --train_n_points ${TRAIN_N_POINTS} \\
            --knn_k ${KNN_K} \\
            --channels ${CHANNELS} \\
            --n_heads ${N_HEADS} \\
            --enc_depth ${ENC_DEPTH} \\
            --dec_depth ${DEC_DEPTH} \\
            --latent_dim ${LATENT_DIM} \\
            --batch_size ${BATCH_SIZE} \\
            --lr ${LR} \\
            --epochs ${EPOCHS} \\
            --warmup_epochs ${WARMUP} \\
            --grad_clip ${GRAD_CLIP} \\
            --wandb_project ${WANDB_PROJECT} \\
            --ckpt_dir ${SLURM_SUBMIT_DIR}/checkpoints \\
            --save_every 10 \\
            --val_every 5 \\
            --amp
" &

wait
echo "Training job $SLURM_JOB_ID finished."
