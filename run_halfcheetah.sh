#!/bin/bash
#SBATCH --job-name=td3_halfcheetah
#SBATCH --output=logs/halfcheetah_%j.out
#SBATCH --error=logs/halfcheetah_%j.err
#SBATCH --partition=L4-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=11:00:00
#SBATCH --mail-user=meshybb@gmail.com
#SBATCH --mail-type=END,FAIL

mkdir -p logs results

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Start: $(date)"
nvidia-smi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

pip install -q mujoco==2.3.7
pip install -q -r requirements.txt

python train.py \
    --env HalfCheetah-v4 \
    --max_timesteps 3000000 \
    --num_runs 3 \
    --batch_size 512

echo "Done: $(date)"
