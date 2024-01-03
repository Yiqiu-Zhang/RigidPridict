#!/bin/bash
#SBATCH --job-name=RigidPredict
#SBATCH -p bio_s1

### This needs to match Trainer(num_nodes=...)
#SBATCH --nodes=1

###单个节点使用的GPU个数，如果不需要GPU，则忽略该项
#SBATCH --gres=gpu:2

### This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=2

#SBATCH --cpus-per-task=12
### request all the memory on a node
#SBATCH --mem=0

#SBATCH --output=out_RigidPredict.log
#SBATCH --error=error_RigidPredict.log

export NCCL_IB_DISABLE=1
export NCCL_IB_HCA=mlx5_0 
export NCCL_SOCKET_IFNAME=eth0
export CUDA_LAUNCH_BLOCKING=1

srun --kill-on-bad-exit=1 python3 train.py /mnt/petrelfs/zhangyiqiu/RigidPridict/bin/train.json --ndevice 2 --node 1 -o Result_RigidPredict