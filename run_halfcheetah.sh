#!/bin/bash
#SBATCH --job-name=td3_halfcheetah
#SBATCH --output=logs/halfcheetah_%j.out
#SBATCH --error=logs/halfcheetah_%j.err
#SBATCH --partition=A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --mail-user=orli.d.p@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:SIGUSR1@120

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

# --- SLURM timeout handler ---
# SLURM sends SIGUSR1 a couple of minutes before the time limit ends
trap "echo 'TIME LIMIT — saving checkpoint'; python train.py --env HalfCheetah-v4 --save_checkpoint_only; exit 0" SIGUSR1

if [ -f results/checkpoint_HalfCheetah-v4.pt ]; then
    echo 'Resuming from checkpoint...'
    python train.py \
        --env HalfCheetah-v4 \
        --max_timesteps 3000000 \
        --num_runs 3 \
        --batch_size 512 \
        --resume results/checkpoint_HalfCheetah-v4.pt
else
    echo 'Starting fresh run...'
    python train.py \
        --env HalfCheetah-v4 \
        --max_timesteps 3000000 \
        --num_runs 3 \
        --batch_size 512
fi

echo "Done: $(date)"
