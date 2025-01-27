#!/bin/bash

#SBATCH --job-name=kfold_melanoma
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=pi_krishnaswamy
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --output=./logs/slurm/kfold/melanoma/%x_%j.out
#SBATCH --error=./logs/slurm/kfold/melanoma/%x_%j.err
cd /home/hm638/PointCloudNet
conda init
conda activate pcenv


python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 4
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 4
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 4
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 4
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 4

python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 1
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 1
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 1
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 1
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 1

python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 2
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 2
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 2
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 2
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 2

python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 8
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 8
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 8
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 8
python main.py --raw_dir melanoma_data_full --full --lr 1e-4 --threshold 0.35 --num_weights 8