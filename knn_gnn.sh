#!/bin/bash

#SBATCH --job-name=knn_res
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --output=./logs/slurm/pdo/%x_%j.out
#SBATCH --error=./logs/slurm/pdo/%x_%j.err
cd /home/hm638/PointCloudNet
conda init
conda activate pcenv

python knn_gnn.py --model GCN
python knn_gnn.py --model SAGE
python knn_gnn.py --model GAT
python knn_gnn.py --model GIN