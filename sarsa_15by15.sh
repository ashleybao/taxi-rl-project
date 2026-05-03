#!/bin/bash
#SBATCH --job-name=bigger_taxi
#SBATCH --output=logs/bigger_taxi_%j.out
#SBATCH --error=logs/bigger_taxi_%j.err

#SBATCH --time=02:00:00
#SBATCH --partition=cpu-q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

# Optional: email notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com

# =========================
# LOAD ENVIRONMENT
# =========================

# If using conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

# =========================
# RUN SCRIPT
# =========================
echo "Starting job at $(date)"
echo "Running on node $(hostname)"

python3 sarsa_15by15.py

echo "Finished at $(date)"