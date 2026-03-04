#!/bin/bash
#===========================================================================
# SLURM Job Script — TACC A100 Nodes (DEPRECATED)
# Rotation-Equivariant Rectified Flow Training
#===========================================================================
#
# NOTE: This script is kept for reference. Use the dedicated scripts:
#   Training:  sbatch scripts/train_tacc.sh
#   Testing:   sbatch scripts/test_tacc.sh
#
# Designed for TACC systems (e.g., Frontera / Lonestar6) with 3x A100 GPUs
# per node and 8 nodes total (24 GPUs).
#===========================================================================

#SBATCH -J rectflow-pc
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=0                      # Use all available memory
#SBATCH -t 24:00:00
#SBATCH -A ASC25079
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ayuj@utexas.edu
#SBATCH --export=ALL

# ------------------------------
# Load system modules
# ------------------------------
module purge
module load cuda/12.2

echo "=========================================="
echo " Job ID:       $SLURM_JOB_ID"
echo " Job Name:     $SLURM_JOB_NAME"
echo " Nodes:        $SLURM_JOB_NODELIST"
echo " Tasks/Node:   $SLURM_NTASKS_PER_NODE"
echo " GPUs/Node:    3"
echo " Total GPUs:   $(( SLURM_NNODES * 3 ))"
echo "=========================================="

# ---- Auto-detect distributed training variables ----
# MASTER_ADDR: first node in the allocation
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# Total number of processes = nodes × GPUs-per-node
NNODES=$SLURM_NNODES
NGPUS_PER_NODE=3
WORLD_SIZE=$(( NNODES * NGPUS_PER_NODE ))

echo " MASTER_ADDR:  $MASTER_ADDR"
echo " MASTER_PORT:  $MASTER_PORT"
echo " WORLD_SIZE:   $WORLD_SIZE"
echo "=========================================="

# ---- Stage data to local scratch for fast I/O ----
# Each node copies data to /tmp so reads bypass the shared filesystem.
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

# ---- Create log directory ----
mkdir -p "${SLURM_SUBMIT_DIR}/logs"
mkdir -p "${SLURM_SUBMIT_DIR}/checkpoints"

# ------------------------------
# Activate conda environment
# ------------------------------
source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
conda activate rectflow
export PYTHONUNBUFFERED=1

# ---- Determine NODE_RANK for this node ----
# srun will launch this script on every node; we compute the node rank
# from the SLURM node list.
NODE_LIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HOSTNAME=$(hostname -s)
NODE_RANK=0
for i in "${!NODE_LIST[@]}"; do
    if [ "${NODE_LIST[$i]}" == "$HOSTNAME" ]; then
        NODE_RANK=$i
        break
    fi
done
export NODE_RANK

echo "Node: $HOSTNAME  |  NODE_RANK: $NODE_RANK"

# ---- Launch training via torchrun (one per node via srun) ----
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
            --n_points 2048 \\
            --channels 128 \\
            --n_heads 8 \\
            --enc_depth 6 \\
            --dec_depth 6 \\
            --latent_dim 256 \\
            --batch_size 32 \\
            --lr 1e-4 \\
            --epochs 300 \\
            --warmup_epochs 10 \\
            --grad_clip 1.0 \\
            --wandb_project rectified-flow-pc \\
            --ckpt_dir ${SLURM_SUBMIT_DIR}/checkpoints \\
            --save_every 10 \\
            --val_every 5 \\
            --amp
" &

wait
echo "Job $SLURM_JOB_ID finished."
