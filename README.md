# DQN on Demon Attack

Deep Reinforcement Learning agent for **Atari Demon Attack**, implemented and experimentally optimized using PyTorch.

The project explores the design, training, and stabilization of a Deep Q-Network through large-scale simulation experiments.

## Highlights

* Implemented a **Deep Q-Network (DQN)** for Atari from scratch in PyTorch
* Added **Double DQN**, **Target Network**, Experience Replay, and Huber Loss
* Conducted **13+ experimental rounds** of hyperparameter and stability analysis
* Trained and evaluated across multiple random seeds
* Executed long-running experiments using **GPU + Slurm**
* Achieved a peak score of **10,593**, exceeding the 9,711 reference benchmark

## Environment

**DemonAttackNoFrameskip-v4**

The preprocessing pipeline includes:

* Grayscale conversion
* 84×84 frame preprocessing
* 4-frame stacking
* Frame skipping
* Reward clipping during training
* Randomized episode initialization

## Architecture

Input `84 × 84 × 4`

→ Conv Layer
→ Conv Layer
→ Fully Connected Layer
→ Q-values for each action

The final implementation uses **Double DQN + Target Network** to improve training stability and reduce Q-value overestimation.

## Experiments

The agent was developed through iterative experimentation with:

* Learning rate
* Replay buffer size
* Batch size
* Exploration schedule
* Target network update frequency
* Optimizer selection
* Training duration

Final experiments were evaluated across **3 independent seeds**.

## Results

| Seed | Peak Score |
| ---- | ---------: |
| 42   |     10,593 |
| 43   |   10,074.5 |
| 44   |      8,146 |

The experiments demonstrated the importance of target networks, Double DQN, replay-buffer diversity, and exploration scheduling for stable RL training.

## Tech Stack

**Python · PyTorch · Gym · OpenCV · NumPy · Slurm · GPU**

## Key Takeaways

This project provided hands-on experience with:

**Reinforcement Learning · Simulation · Algorithm Design · GPU Experimentation · Hyperparameter Optimization · Statistical Evaluation**
