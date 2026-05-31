Files Created:
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

How to Run

pip install -r requirements.txt

# InvertedPendulum (≈1h on CPU, much faster on GPU)
python train.py --env InvertedPendulum-v4

# HalfCheetah (longer — 3M steps)
python train.py --env HalfCheetah-v4 --max_timesteps 3000000

# Record video for submission
python record_video.py --env InvertedPendulum-v4 \
    --ckpt results/InvertedPendulum-v4_run1_seed42.pt


Hyperparameters (vs. paper)

| Parameter | Paper | This code | Note |
|---|---|---|---|
| LR (actor/critic) | 1e-3 | 3e-4 | changed |
| Hidden dim | 400/300 | 256/256 | changed |
| Batch size | 100 | 256 | changed |
| τ (soft update) | 0.005 | 0.005 | same |
| Policy noise σ | 0.2 | 0.2 | same |
| Policy delay d | 2 | 2 | same |
| Discount γ | 0.99 | 0.99 | same |

