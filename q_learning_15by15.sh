#! /bin/sh
#SBATCH --job-name=q-learning-15by15
#SBATCH --partition cpu-q
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --output ./logs/q-learning-15by15-%j.out
#SBATCH --error ./logs/q-learning-15by15-%j.err

module purge
module load cuda11.8/toolkit

PYTHONNOUSERSITE=1 conda run -p ./conda-taxi python q_learning_15by15.py