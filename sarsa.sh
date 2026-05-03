#! /bin/sh
#SBATCH --job-name=sarsa
#SBATCH --partition cpu-q
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --output ./logs/sarsa-%j.out
#SBATCH --error ./logs/sarsa-%j.err

module purge
module load cuda11.8/toolkit

PYTHONNOUSERSITE=1 conda run -p ./conda-taxi python sarsa.py