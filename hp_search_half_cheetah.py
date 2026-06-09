import gymnasium as gym
import numpy as np
import json
import os

from td3_agent import TD3, ReplayBuffer

ENV_NAME = "HalfCheetah-v4"
MAX_STEPS = 500000  # Extended horizon to test convergence stability
START_STEPS = 10000
EVAL_FREQ = 10000
EVAL_EPISODES = 10

# Testing only the Top 3 configurations
configs_to_test = [
    {"name": "Baseline_Paper", "lr": 1e-3, "batch_size": 100, "hidden_dim": 400},
    {"name": "Our_Chosen_Config", "lr": 3e-4, "batch_size": 512, "hidden_dim": 256},
    {"name": "Hybrid_Fast_LR_Large_Batch", "lr": 1e-3, "batch_size": 512, "hidden_dim": 256}
]

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
            noise = np.random.normal(0, max_action * 0.1, size=action_dim)
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
            print(f"[{config['name']}] Step: {t+1} | Avg Reward: {avg_reward:.2f}")
            evaluations.append({"step": t+1, "reward": avg_reward})
            
    env.close()
    return evaluations

if __name__ == "__main__":
    results = {}
    print(f"Starting Extended HP Search for {ENV_NAME}")
    
    for i, config in enumerate(configs_to_test):
        print(f"\n--- Running Config {i+1}/{len(configs_to_test)}: {config['name']} ---")
        eval_curve = run_experiment(config)
        results[config['name']] = eval_curve
        
    output_file = f"hp_search_results_extended_{ENV_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nExtended search complete! Results saved to {output_file}")