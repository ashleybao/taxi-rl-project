#! /bin/sh
#SBATCH --job-name=neural_fa
#SBATCH --partition cpu-q
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --output ./logs/neural-fa-%j.out
#SBATCH --error ./logs/neural-fa-%j.err

module purge
module load cuda11.8/toolkit

PYTHONNOUSERSITE=1 conda run -p ./conda-taxi python neural_fa.py