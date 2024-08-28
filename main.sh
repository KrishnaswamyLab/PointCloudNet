#!/bin/bash

#SBATCH --job-name=test_main
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err
cd ~/gpfs/gibbs/pi/krishnaswamy_smita/sv496/PointCloudNet/
module load miniconda
conda activate mfcn

python main.py --raw_dir melanoma_data