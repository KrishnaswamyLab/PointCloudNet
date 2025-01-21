#!/bin/bash

#SBATCH --job-name=knn_res
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=pi_krishnaswamy
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --output=./logs/slurm/knn/%x_%j.out
#SBATCH --error=./logs/slurm/knn/%x_%j.err
cd /home/hm638/PointCloudNet
conda init
conda activate pcenv

python knn_gnn.py --model GCN
python knn_gnn.py --model SAGE
python knn_gnn.py --model GAT
python knn_gnn.py --model GIN

python knn_gnn.py --model GCN --raw_dir melanoma_data_full
python knn_gnn.py --model SAGE --raw_dir melanoma_data_full
python knn_gnn.py --model GAT --raw_dir melanoma_data_full
python knn_gnn.py --model GIN --raw_dir melanoma_data_full

python knn_gnn.py --model GCN --raw_dir melanoma_data_full --full
python knn_gnn.py --model SAGE --raw_dir melanoma_data_full --full
python knn_gnn.py --model GAT --raw_dir melanoma_data_full --full
python knn_gnn.py --model GIN --raw_dir melanoma_data_full --full