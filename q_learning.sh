#! /bin/sh
#SBATCH --job-name=qlearn
#SBATCH --partition cpu-q
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --output ./logs/qlearn-%j.out
#SBATCH --error ./logs/qlearn-%j.err

module purge
module load cuda11.8/toolkit

PYTHONNOUSERSITE=1 conda run -p ./conda-taxi python q_learning.py