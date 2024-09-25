#!/usr/bin/env bash
#SBATCH --time=120:00:00
#SBATCH --partition=gpu_h100
#SBATCH --exclusive
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

set -e
set -x

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_NTASKS: $SLURM_NTASKS"

module load 2023
module load CUDA/12.1.1

eval "$(conda shell.bash hook)"
conda activate openmmlab
echo "Activated conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Set environment variables for torch.distributed
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$((10000 + $RANDOM % 10000))

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Set NCCL and Gloo to use the Ethernet interface for initialization
export NCCL_SOCKET_IFNAME=eno2np0
export GLOO_SOCKET_IFNAME=eno2np0
echo "NCCL_SOCKET_IFNAME set to $NCCL_SOCKET_IFNAME"
echo "GLOO_SOCKET_IFNAME set to $GLOO_SOCKET_IFNAME"

# Retain NCCL debugging for verification
export NCCL_DEBUG=INFO

# Run the training script
srun --export=ALL \
    --kill-on-bad-exit=1 \
    python train.py /home/tkerssies/ViT-Adapter/detection/configs/mask2former/mask2former_beitv2_adapter_large_16x1_3x_coco-panoptic.py \
    --work-dir=. \
    --seed 0