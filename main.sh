#!/bin/bash

#SBATCH --job-name=test_main
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err
cd /home/hm638/PointCloudNet
conda init
conda activate pcenv

python main.py --num_weights 4
python main.py --num_weights 3
python main.py --num_weights 2
python main.py --num_weights 1