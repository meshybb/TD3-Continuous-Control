"""
Training script for TD3 on MuJoCo environments (Single Seed, Slurm Array ready).

Usage:
  python train.py --env InvertedPendulum-v4 --seed 42
  python train.py --env HalfCheetah-v4 --max_timesteps 3000000 --seed 55 --resume
"""

import argparse
import json
import os
import time
import pickle # For saving replay buffer state

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym

from td3_agent import TD3, ReplayBuffer

ENV_DEFAULTS = {
    "InvertedPendulum-v4": {"max_timesteps": 1_000_000, "start_timesteps": 5_000},
    "HalfCheetah-v4":      {"max_timesteps": 3_000_000, "start_timesteps": 10_000},
}

def evaluate_policy(policy: TD3, env_name: str, seed: int, eval_episodes: int = 10) -> float:
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

def train_td3_single_run(
    env_name:        str,
    seed:            int,
    max_timesteps:   int,
    start_timesteps: int,
    eval_freq:       int,
    batch_size:      int,
    discount:        float,
    tau:             float,
    policy_noise:    float,
    noise_clip:      float,
    policy_freq:     int,
    expl_noise:      float,
    actor_lr:        float,
    critic_lr:       float,
    hidden_dim:      int,
    eval_episodes:   int,
    save_dir:        str,
    resume:          bool = False
):
    env = gym.make(env_name)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

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
    best_eval_reward = -float("inf")
    start_t = 1

    # --- Real Resume Logic ---
    state_file = os.path.join(save_dir, f"{env_name}_seed{seed}_state.pt")
    if resume and os.path.exists(state_file):
        print(f"Loading checkpoint from {state_file}...")
        checkpoint = torch.load(state_file, weights_only=False)
        
        # Load Networks
        policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        policy.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        policy.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        
        # Load Optimizers (Fixed variable names)
        policy.actor_opt.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        policy.critic_opt.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        start_t = checkpoint['step'] + 1
        eval_rewards = checkpoint['eval_rewards']
        eval_steps = checkpoint['eval_steps']
        best_eval_reward = checkpoint.get('best_eval_reward', -float("inf"))
        
        # Load Replay Buffer
        buffer_file = os.path.join(save_dir, f"{env_name}_seed{seed}_buffer.pkl")
        if os.path.exists(buffer_file):
             with open(buffer_file, 'rb') as f:
                 loaded_buffer = pickle.load(f)
                 replay_buffer.states = loaded_buffer['states']
                 replay_buffer.actions = loaded_buffer['actions']
                 replay_buffer.next_states = loaded_buffer['next_states']
                 replay_buffer.rewards = loaded_buffer['rewards']
                 replay_buffer.dones = loaded_buffer['dones']
                 replay_buffer.ptr = loaded_buffer['ptr']
                 replay_buffer.size = loaded_buffer['size']
                 print(f"Replay buffer loaded (size: {replay_buffer.size}). Resuming from step {start_t}.")
        else:
             print("Warning: Checkpoint found but replay buffer missing. Starting with empty buffer.")
    # --------------------------------

    state, _ = env.reset(seed=seed)
    ep_reward = ep_steps = ep_num = 0
    t0 = time.time()

    for t in range(start_t, max_timesteps + 1):
        ep_steps += 1

        if t <= start_timesteps:
            action = env.action_space.sample()
        else:
            noise  = np.random.normal(0, max_action * expl_noise, size=action_dim)
            action = (policy.select_action(np.array(state)) + noise).clip(-max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward

        done_bool = float(terminated)
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state

        if t > start_timesteps:
            policy.train(replay_buffer, batch_size)

        if terminated or truncated:
            print(f"[{env_name}] seed={seed:3d} t={t:>9,}  ep={ep_num+1:>4}  ep_reward={ep_reward:>9.2f}  ep_steps={ep_steps}")
            state, _ = env.reset()
            ep_reward = ep_steps = 0
            ep_num += 1

        if t % eval_freq == 0:
            avg_r   = evaluate_policy(policy, env_name, seed, eval_episodes)
            elapsed = (time.time() - t0) / 60.0
            eval_rewards.append(avg_r)
            eval_steps.append(t)
            print(f"  *** Eval  t={t:>9,}  avg_reward={avg_r:>9.2f}  ({eval_episodes} eps)  elapsed={elapsed:.1f}min ***")

            os.makedirs(save_dir, exist_ok=True)
            
            # Save Policy Network (.pt)
            ckpt_path = os.path.join(save_dir, f"{env_name}_seed{seed}.pt")
            policy.save(ckpt_path)
            
            # Save Best Policy Network
            if avg_r > best_eval_reward:
                best_eval_reward = avg_r
                best_ckpt_path = os.path.join(save_dir, f"{env_name}_seed{seed}_best.pt")
                policy.save(best_ckpt_path)

            # --- Save Full State Checkpoint for Resuming ---
            torch.save({
                'step': t,
                'actor_state_dict': policy.actor.state_dict(),
                'critic_state_dict': policy.critic.state_dict(),
                'actor_target_state_dict': policy.actor_target.state_dict(),
                'critic_target_state_dict': policy.critic_target.state_dict(),
                'actor_optimizer_state_dict': policy.actor_opt.state_dict(), # Fixed variable name
                'critic_optimizer_state_dict': policy.critic_opt.state_dict(), # Fixed variable name
                'eval_rewards': eval_rewards,
                'eval_steps': eval_steps,
                'best_eval_reward': best_eval_reward
            }, state_file)
            
            # Save Replay Buffer
            buffer_file = os.path.join(save_dir, f"{env_name}_seed{seed}_buffer.pkl")
            with open(buffer_file, 'wb') as f:
                pickle.dump({
                    'states': replay_buffer.states,
                    'actions': replay_buffer.actions,
                    'next_states': replay_buffer.next_states,
                    'rewards': replay_buffer.rewards,
                    'dones': replay_buffer.dones,
                    'ptr': replay_buffer.ptr,
                    'size': replay_buffer.size
                }, f)
            # -----------------------------------------------------

    env.close()

    # Save final results JSON for THIS seed
    results_out = {
        "env": args.env,
        "seed": seed,
        "final_reward": eval_rewards[-1] if eval_rewards else -float("inf"),
        "eval_rewards": eval_rewards,
        "eval_steps": eval_steps,
    }
    rpath = os.path.join(save_dir, f"{env_name}_seed{seed}_results.json")
    with open(rpath, "w") as f:
        json.dump(results_out, f, indent=2)

def main():
    global args
    parser = argparse.ArgumentParser(description="Train TD3 on a MuJoCo v4 environment")

    parser.add_argument("--env",             default="InvertedPendulum-v4")
    parser.add_argument("--max_timesteps",   type=int,   default=None)
    parser.add_argument("--start_timesteps", type=int,   default=None)
    parser.add_argument("--eval_freq",       type=int,   default=10_000)
    parser.add_argument("--eval_episodes",   type=int,   default=10)
    parser.add_argument("--batch_size",      type=int,   default=256)
    parser.add_argument("--discount",        type=float, default=0.99)
    parser.add_argument("--tau",             type=float, default=0.005)
    parser.add_argument("--policy_noise",    type=float, default=0.2)
    parser.add_argument("--noise_clip",      type=float, default=0.5)
    parser.add_argument("--policy_freq",     type=int,   default=2)
    parser.add_argument("--expl_noise",      type=float, default=0.1)
    parser.add_argument("--actor_lr",        type=float, default=3e-4)
    parser.add_argument("--critic_lr",       type=float, default=3e-4)
    parser.add_argument("--hidden_dim",      type=int,   default=256)
    parser.add_argument("--seed",            type=int,   required=True)
    parser.add_argument("--save_dir",        default="results")
    parser.add_argument("--resume",          action="store_true", help="Resume from checkpoint if exists")
    args = parser.parse_args()

    # Fill env-specific defaults
    defaults = ENV_DEFAULTS.get(args.env, {"max_timesteps": 1_000_000, "start_timesteps": 10_000})
    if args.max_timesteps   is None: args.max_timesteps   = defaults["max_timesteps"]
    if args.start_timesteps is None: args.start_timesteps = defaults["start_timesteps"]

    train_td3_single_run(
        env_name        = args.env,
        seed            = args.seed,
        max_timesteps   = args.max_timesteps,
        start_timesteps = args.start_timesteps,
        eval_freq       = args.eval_freq,
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
        eval_episodes   = args.eval_episodes,
        save_dir        = args.save_dir,
        resume          = args.resume
    )

if __name__ == "__main__":
    main()
