import gymnasium as gym
import numpy as np
import json
import os
from itertools import product

from td3_agent import TD3, ReplayBuffer

ENV_NAME = "HalfCheetah-v4"
MAX_STEPS = 200000  
START_STEPS = 10000
EVAL_FREQ = 10000
EVAL_EPISODES = 10

# Hyperparameter grid specifically testing your changes vs the TD3 paper
param_grid = {
    "lr": [1e-3, 3e-4],          # Paper: 1e-3 | Yours: 3e-4
    "batch_size": [100, 512],    # Paper: 100  | Yours: 512
    "hidden_dim": [400, 256]     # Paper: 400 (approx) | Yours: 256
}

def evaluate_policy(env_name, agent, eval_episodes):
    eval_env = gym.make(env_name)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.select_action(np.array(state))
            state, reward, done, truncated, _ = eval_env.step(action)
            avg_reward += reward
            
    avg_reward /= eval_episodes
    eval_env.close()
    return avg_reward

def run_experiment(config):
    env = gym.make(ENV_NAME)
    state, _ = env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=config['lr'],
        critic_lr=config['lr'],
        hidden_dim=config['hidden_dim']
    )
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    evaluations = []
    
    for t in range(int(MAX_STEPS)):
        if t < START_STEPS:
            action = env.action_space.sample()
        else:
            expl_noise = 0.1 # Default from paper
            noise = np.random.normal(0, max_action * expl_noise, size=action_dim)
            action = (agent.select_action(np.array(state)) + noise).clip(-max_action, max_action)
            
        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer.add(state, action, next_state, reward, float(done))
        state = next_state
        
        if t >= START_STEPS:
            agent.train(replay_buffer, batch_size=config['batch_size'])
            
        if done or truncated:
            state, _ = env.reset()
            
        if (t + 1) % EVAL_FREQ == 0:
            avg_reward = evaluate_policy(ENV_NAME, agent, EVAL_EPISODES)
            print(f"Step: {t+1} | Config: {config} | Avg Reward: {avg_reward:.2f}")
            evaluations.append({"step": t+1, "reward": avg_reward})
            
    env.close()
    return evaluations

if __name__ == "__main__":
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]
    
    results = {}
    
    print(f"Starting Hyperparameter Search for {ENV_NAME}")
    print(f"Total combinations: {len(experiments)}")
    
    for i, config in enumerate(experiments):
        print(f"\n--- Running Experiment {i+1}/{len(experiments)} ---")
        print(f"Parameters: {config}")
        
        eval_curve = run_experiment(config)
        
        config_str = f"lr={config['lr']}_bs={config['batch_size']}_hd={config['hidden_dim']}"
        results[config_str] = eval_curve
        
    output_file = f"hp_search_results_{ENV_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSearch complete! Results saved to {output_file}")