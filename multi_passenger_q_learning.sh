#! /bin/sh
#SBATCH --job-name=multi-q
#SBATCH --partition cpu-q
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --output ./logs/multi-q-%j.out
#SBATCH --error ./logs/multi-q-%j.err

module purge

PYTHONNOUSERSITE=1 conda run -p ./conda-taxi python multi_passenger_q_learning.py