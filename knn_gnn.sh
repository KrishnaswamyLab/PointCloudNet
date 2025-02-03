#!/bin/bash

#SBATCH --job-name=knn_res
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --output=./logs/slurm/persistence/baseline/%x_%j.out
#SBATCH --error=./logs/slurm/persistence/baseline/%x_%j.err
cd /home/hm638/PointCloudNet
conda init
conda activate pcenv

python knn_gnn.py --model GCN --raw_dir pdo_data
python knn_gnn.py --model SAGE --raw_dir pdo_data
python knn_gnn.py --model GAT --raw_dir pdo_data
python knn_gnn.py --model GIN --raw_dir pdo_data
