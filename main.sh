#!/bin/bash

#SBATCH --job-name=melanoma_ablation
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --output=./logs/slurm/melanoma/%x_%j.out
#SBATCH --error=./logs/slurm/melanome/%x_%j.err
cd /home/hm638/PointCloudNet
conda init
conda activate pcenv


python main.py --num_weights 16
python main.py --num_weights 8
python main.py --num_weights 4
python main.py --num_weights 2
python main.py --num_weights 1