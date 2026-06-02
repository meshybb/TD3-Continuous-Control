"""
Training script for TD3 on MuJoCo environments.

Usage:
  python train.py --env InvertedPendulum-v4
  python train.py --env HalfCheetah-v4 --max_timesteps 3000000
  python train.py --env InvertedPendulum-v4 --num_runs 3 --eval_episodes 10
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym

from td3_agent import TD3, ReplayBuffer


# --------------------------------------------------------------------------- #
#  Environment-specific defaults (deliberately differ from the paper)         #
# --------------------------------------------------------------------------- #
ENV_DEFAULTS = {
    "InvertedPendulum-v4": {"max_timesteps": 1_000_000, "start_timesteps": 5_000},
    "HalfCheetah-v4":      {"max_timesteps": 3_000_000, "start_timesteps": 10_000},
}


# --------------------------------------------------------------------------- #
#  Evaluation                                                                 #
# --------------------------------------------------------------------------- #
def evaluate_policy(
    policy: TD3,
    env_name: str,
    seed: int,
    eval_episodes: int = 10,
) -> float:
    """Run deterministic (no exploration noise) evaluation episodes."""
    env = gym.make(env_name)
    total = 0.0
    for ep in range(eval_episodes):
        state, _ = env.reset(seed=seed + 1000 + ep)
        terminated = truncated = False
        while not terminated and not truncated:
            action = policy.select_action(np.array(state))
            state, r, terminated, truncated, _ = env.step(action)
            total += r
    env.close()
    return total / eval_episodes


# --------------------------------------------------------------------------- #
#  Single training run                                                        #
# --------------------------------------------------------------------------- #
def train_td3(
    env_name:        str,
    seed:            int,
    max_timesteps:   int   = 1_000_000,
    start_timesteps: int   = 10_000,
    eval_freq:       int   = 10_000,
    batch_size:      int   = 256,
    discount:        float = 0.99,
    tau:             float = 0.005,
    policy_noise:    float = 0.2,
    noise_clip:      float = 0.5,
    policy_freq:     int   = 2,
    expl_noise:      float = 0.1,
    actor_lr:        float = 3e-4,
    critic_lr:       float = 3e-4,
    hidden_dim:      int   = 256,
    eval_episodes:   int   = 10,
    save_dir:        str   = "results",
    run_id:          int   = 0,
):
    """Train a TD3 agent for one run and return the evaluation history."""
    env = gym.make(env_name)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Hyperparameters: policy_noise and noise_clip are fractions of max_action
    policy = TD3(
        state_dim    = state_dim,
        action_dim   = action_dim,
        max_action   = max_action,
        discount     = discount,
        tau          = tau,
        policy_noise = policy_noise * max_action,
        noise_clip   = noise_clip   * max_action,
        policy_freq  = policy_freq,
        actor_lr     = actor_lr,
        critic_lr    = critic_lr,
        hidden_dim   = hidden_dim,
    )
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    eval_rewards = []
    eval_steps   = []
    
    # --- ADDED: Track best evaluation reward for checkpointing ---
    best_eval_reward = -float("inf")

    state, _ = env.reset(seed=seed)
    ep_reward = ep_steps = ep_num = 0
    t0 = time.time()

    for t in range(1, max_timesteps + 1):
        ep_steps += 1

        # Select action: random warm-up, then policy + Gaussian noise
        if t <= start_timesteps:
            action = env.action_space.sample()
        else:
            noise  = np.random.normal(0, max_action * expl_noise, size=action_dim)
            action = (policy.select_action(np.array(state)) + noise).clip(
                -max_action, max_action
            )

        next_state, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward

        # done_bool = 1 only on true termination, not time-limit truncation.
        # This ensures we correctly bootstrap through timeout boundaries.
        done_bool = float(terminated)
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state

        if t > start_timesteps:
            policy.train(replay_buffer, batch_size)

        if terminated or truncated:
            print(
                f"[{env_name}] run={run_id+1} seed={seed:3d} "
                f"t={t:>9,}  ep={ep_num+1:>4}  "
                f"ep_reward={ep_reward:>9.2f}  ep_steps={ep_steps}"
            )
            state, _ = env.reset()
            ep_reward = ep_steps = 0
            ep_num += 1

        # Evaluate every eval_freq steps (no exploration noise)
        if t % eval_freq == 0:
            avg_r   = evaluate_policy(policy, env_name, seed, eval_episodes)
            elapsed = (time.time() - t0) / 60.0
            eval_rewards.append(avg_r)
            eval_steps.append(t)
            print(
                f"  *** Eval  t={t:>9,}  "
                f"avg_reward={avg_r:>9.2f}  "
                f"({eval_episodes} eps)  "
                f"elapsed={elapsed:.1f}min ***"
            )

            # --- ADDED: Checkpointing during evaluation ---
            os.makedirs(save_dir, exist_ok=True)
            
            # 1. Save the latest model (overwrites previous to save space)
            ckpt_path = os.path.join(save_dir, f"{env_name}_run{run_id+1}_seed{seed}.pt")
            policy.save(ckpt_path)
            
            # 2. Save the best model seen so far
            if avg_r > best_eval_reward:
                best_eval_reward = avg_r
                best_ckpt_path = os.path.join(save_dir, f"{env_name}_run{run_id+1}_seed{seed}_best.pt")
                policy.save(best_ckpt_path)
                print(f"  [Checkpoint] Saved new best model (Reward: {avg_r:.2f})")
            # ----------------------------------------------

    env.close()

    # The final model is already saved during the last evaluation step.
    return eval_rewards, eval_steps, policy


# --------------------------------------------------------------------------- #
#  Multi-run experiment                                                       #
# --------------------------------------------------------------------------- #
def run_experiment(env_name, seeds, save_dir="results", **kwargs):
    all_rewards = []
    all_steps   = None
    best_policy = None
    best_final  = -float("inf")

    for run_id, seed in enumerate(seeds):
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  {env_name}  |  Run {run_id+1}/{len(seeds)}  |  Seed {seed}")
        print(sep)

        rewards, steps, policy = train_td3(
            env_name = env_name,
            seed     = seed,
            save_dir = save_dir,
            run_id   = run_id,
            **kwargs,
        )
        all_rewards.append(rewards)
        if all_steps is None:
            all_steps = steps

        final = rewards[-1] if rewards else -float("inf")
        if final > best_final:
            best_final, best_policy = final, policy

        print(f"\n  Run {run_id+1} final eval reward: {final:.2f}")

    finals = [r[-1] for r in all_rewards]
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {env_name}  |  Summary over {len(seeds)} runs")
    print(f"  Final rewards: {[f'{r:.1f}' for r in finals]}")
    print(f"  Mean ± Std   : {np.mean(finals):.2f} ± {np.std(finals):.2f}")
    print(sep)

    return all_rewards, all_steps, best_policy, finals


# --------------------------------------------------------------------------- #
#  Plotting                                                                   #
# --------------------------------------------------------------------------- #
def plot_results(all_rewards, all_steps, env_name, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["royalblue", "darkorange", "forestgreen"]

    for i, rewards in enumerate(all_rewards):
        ax.plot(all_steps, rewards, color=colors[i % len(colors)],
                linewidth=1.5, label=f"Run {i+1}", alpha=0.9)

    if len(all_rewards) > 1:
        mean_r = np.mean(all_rewards, axis=0)
        ax.plot(all_steps, mean_r, color="crimson", linewidth=2.5,
                linestyle="--", label="Mean")

    ax.set_xlabel("Training Steps",    fontsize=13)
    ax.set_ylabel("Evaluation Reward", fontsize=13)
    ax.set_title(f"TD3 — {env_name}",  fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, f"{env_name}_learning_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path}")


# --------------------------------------------------------------------------- #
#  Entry point                                                                #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Train TD3 on a MuJoCo v4 environment")

    parser.add_argument("--env",             default="InvertedPendulum-v4",
                        help="Gymnasium MuJoCo environment ID (v4)")
    parser.add_argument("--max_timesteps",   type=int,   default=None,
                        help="Total env.step() calls (default: env-specific)")
    parser.add_argument("--start_timesteps", type=int,   default=None,
                        help="Random warm-up steps before policy is used")
    parser.add_argument("--eval_freq",       type=int,   default=10_000)
    parser.add_argument("--eval_episodes",   type=int,   default=10)
    parser.add_argument("--batch_size",      type=int,   default=256)
    parser.add_argument("--discount",        type=float, default=0.99)
    parser.add_argument("--tau",             type=float, default=0.005)
    parser.add_argument("--policy_noise",    type=float, default=0.2,
                        help="Target policy noise std (as fraction of max_action)")
    parser.add_argument("--noise_clip",      type=float, default=0.5,
                        help="Target policy noise clip (as fraction of max_action)")
    parser.add_argument("--policy_freq",     type=int,   default=2)
    parser.add_argument("--expl_noise",      type=float, default=0.1,
                        help="Exploration noise std (as fraction of max_action)")
    parser.add_argument("--actor_lr",        type=float, default=3e-4)
    parser.add_argument("--critic_lr",       type=float, default=3e-4)
    parser.add_argument("--hidden_dim",      type=int,   default=256)
    parser.add_argument("--num_runs",        type=int,   default=3)
    parser.add_argument("--save_dir",        default="results")
    args = parser.parse_args()

    # Fill env-specific defaults when not provided on the command line
    defaults = ENV_DEFAULTS.get(
        args.env, {"max_timesteps": 1_000_000, "start_timesteps": 10_000}
    )
    if args.max_timesteps   is None: args.max_timesteps   = defaults["max_timesteps"]
    if args.start_timesteps is None: args.start_timesteps = defaults["start_timesteps"]

    # Seeds are spread out to ensure diverse coverage
    seeds = [42 + i * 13 for i in range(args.num_runs)]

    hyperparams = dict(
        max_timesteps   = args.max_timesteps,
        start_timesteps = args.start_timesteps,
        eval_freq       = args.eval_freq,
        eval_episodes   = args.eval_episodes,
        batch_size      = args.batch_size,
        discount        = args.discount,
        tau             = args.tau,
        policy_noise    = args.policy_noise,
        noise_clip      = args.noise_clip,
        policy_freq     = args.policy_freq,
        expl_noise      = args.expl_noise,
        actor_lr        = args.actor_lr,
        critic_lr       = args.critic_lr,
        hidden_dim      = args.hidden_dim,
    )

    print("\nHyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k:20s} = {v}")
    print(f"  {'seeds':20s} = {seeds}")
    print()

    all_rewards, all_steps, best_policy, finals = run_experiment(
        env_name = args.env,
        seeds    = seeds,
        save_dir = args.save_dir,
        **hyperparams,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    plot_results(all_rewards, all_steps, args.env, args.save_dir)

    # Save full results to JSON for later analysis / report
    results_out = {
        "env":             args.env,
        "seeds":           seeds,
        "final_rewards":   finals,
        "mean":            float(np.mean(finals)),
        "std":             float(np.std(finals)),
        "all_rewards":     all_rewards,
        "all_steps":       all_steps,
        "hyperparameters": hyperparams,
    }
    rpath = os.path.join(args.save_dir, f"{args.env}_results.json")
    with open(rpath, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"Full results saved → {rpath}")


if __name__ == "__main__":
    main()