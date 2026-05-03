#! /bin/sh
#SBATCH --job-name=linear-fa
#SBATCH --partition cpu-q
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --output ./logs/linear-fa-%j.out
#SBATCH --error ./logs/linear-fa-%j.err

module purge
module load cuda11.8/toolkit

PYTHONNOUSERSITE=1 conda run -p ./conda-taxi python linear_fa.py