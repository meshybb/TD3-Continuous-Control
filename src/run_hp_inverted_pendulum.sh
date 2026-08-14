#!/bin/bash
#SBATCH --job-name=hp_td3_pendulum
#SBATCH --output=logs/hp_pendulum_%j.out
#SBATCH --error=logs/hp_pendulum_%j.err
#SBATCH --partition=A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --mail-user=orli.d.p@gmail.com
#SBATCH --mail-type=END,FAIL

mkdir -p logs

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi

# מניעת פתיחת מספר מוגזם של תהליכוני CPU על ידי PyTorch
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# התקנת סביבות ותלויות
pip install -q mujoco==2.3.7
pip install -q -r requirements.txt

# הרצת סקריפט החיפוש
python hp_search_inverted_pendulum.py

echo "Done: $(date)"