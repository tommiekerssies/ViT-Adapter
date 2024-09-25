#!/usr/bin/env bash
#SBATCH --time=120:00:00
#SBATCH --partition=gpu_h100
#SBATCH --exclusive
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

module load 2023
module load CUDA/12.1.1

eval "$(conda shell.bash hook)"
conda activate openmmlab

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$((10000 + $RANDOM % 10000))

export NCCL_SOCKET_IFNAME=eno2np0
export GLOO_SOCKET_IFNAME=eno2np0

srun --export=ALL \
    --kill-on-bad-exit=1 \
    python train.py /home/tkerssies/ViT-Adapter/detection/configs/mask2former/mask2former_beitv2_adapter_large_16x1_3x_coco-panoptic.py \
    --work-dir=. \
    --seed 0