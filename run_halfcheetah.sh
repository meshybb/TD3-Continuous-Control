#!/bin/bash
#SBATCH --job-name=td3_halfcheetah
#SBATCH --output=logs/halfcheetah_%A_%a.out  # %A = Job ID, %a = Array index
#SBATCH --error=logs/halfcheetah_%A_%a.err
#SBATCH --partition=A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --mail-user=orli.d.p@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --array=0-2  # This creates 3 jobs running in parallel!

mkdir -p logs results

# Define seeds 55 68
SEEDS=(42)
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Seed for this run: $CURRENT_SEED"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Start: $(date)"
nvidia-smi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

pip install -q mujoco==2.3.7
pip install -q -r requirements.txt

# --- SLURM timeout handler ---
# No need to explicitly call save here, because our train.py saves every eval_freq.
# SLURM will kill it gracefully, and on requeue it will resume from the last eval.
trap "echo 'TIME LIMIT — caught SIGUSR1'; exit 0" SIGUSR1

# The --resume flag now natively handles checking if a state file exists in train.py
python train.py \
    --env HalfCheetah-v4 \
    --max_timesteps 3000000 \
    --batch_size 512 \
    --seed $CURRENT_SEED \
    --resume

echo "Done: $(date)"