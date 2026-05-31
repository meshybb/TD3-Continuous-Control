"""
Record video of a trained TD3 agent.

Requires ffmpeg to be installed (conda install ffmpeg  OR  apt install ffmpeg).

Usage:
  python record_video.py --env InvertedPendulum-v4 \\
      --ckpt results/InvertedPendulum-v4_run1_seed42.pt \\
      --episodes 3 --out_dir videos
"""

import argparse
import os

import numpy as np
import gymnasium as gym

from td3_agent import TD3


def record_video(
    env_name:     str,
    ckpt_path:    str,
    out_dir:      str = "videos",
    num_episodes: int = 3,
    seed:         int = 0,
):
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder    = out_dir,
        episode_trigger = lambda ep: True,   # record every episode
        name_prefix     = env_name,
    )

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(state_dim, action_dim, max_action)
    policy.load(ckpt_path)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Recording {num_episodes} episode(s) to {out_dir}/\n")

    total_reward = 0.0
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        ep_reward  = 0.0
        while not terminated and not truncated:
            action = policy.select_action(np.array(state))
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
        total_reward += ep_reward
        print(f"  Episode {ep+1}: reward = {ep_reward:.2f}")

    env.close()
    print(f"\nMean reward: {total_reward / num_episodes:.2f}")
    print(f"Videos saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",      required=True,
                        help="e.g. InvertedPendulum-v4 or HalfCheetah-v4")
    parser.add_argument("--ckpt",     required=True,
                        help="Path to .pt checkpoint file from train.py")
    parser.add_argument("--out_dir",  default="videos")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed",     type=int, default=0)
    args = parser.parse_args()

    record_video(args.env, args.ckpt, args.out_dir, args.episodes, args.seed)


if __name__ == "__main__":
    main()
