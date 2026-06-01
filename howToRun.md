Files:
td3_agent.py
Core TD3 implementation with four classes:

ReplayBuffer — pre-allocated numpy circular buffer, samples mini-batches as GPU tensors
Actor — 2-layer MLP (256→256), tanh output scaled by max_action
Critic — twin Q-networks (Q1, Q2) that take [state, action] concatenated at input
TD3 — implements all three paper contributions:
Clipped Double Q-learning: y = r + γ · min(Q1_target, Q2_target)
Target policy smoothing: ã = clip(π_target(s') + clip(ε, -c, c), ±max_action)
Delayed policy updates: actor + targets updated every policy_freq=2 critic steps
train.py
Full training pipeline:

Evaluate every 10,000 steps (no exploration noise) — exactly as required
Runs 3 seeds [42, 55, 68] per environment automatically
Saves per-run .pt checkpoints, learning curve PNG, and full results.json
Environment-specific defaults: 1M steps for InvertedPendulum, 3M for HalfCheetah
record_video.py
Records video of trained policy using gymnasium.wrappers.RecordVideo (requires ffmpeg).

run_inverted_pendulum.sh
SLURM job script for InvertedPendulum-v4 — requests 1 A100 GPU, 8 CPUs, 32G RAM, 3.5h time limit.
Installs dependencies, runs 3 seeds, saves results to results/ and logs to logs/.

run_halfcheetah.sh
SLURM job script for HalfCheetah-v4 — requests 1 L4 GPU, 8 CPUs, 32G RAM, 11h time limit.
Installs dependencies, runs 3 seeds, saves results to results/ and logs to logs/.

requirements.txt
Python dependencies: torch, numpy, gymnasium[mujoco], mujoco==2.3.7 (pinned for Python 3.9 wheel), matplotlib.

How to Run (SLURM cluster)

# Step 1 — install dependencies (run once on the login node)
pip install -r requirements.txt

# Step 2 — submit training jobs
sbatch run_inverted_pendulum.sh   # A100 GPU, up to 3.5h
sbatch run_halfcheetah.sh         # L4 GPU, up to 11h

# Monitor your jobs
squeue -u $USER

# Logs are written to logs/ while the job runs
tail -f logs/inverted_pendulum_<JOBID>.out
tail -f logs/halfcheetah_<JOBID>.out

# Step 3 — record video after training finishes (run on login node)
pip install -q mujoco==2.3.7
python record_video.py --env InvertedPendulum-v4 \
    --ckpt results/InvertedPendulum-v4_run1_seed42.pt

python record_video.py --env HalfCheetah-v4 \
    --ckpt results/HalfCheetah-v4_run1_seed42.pt




