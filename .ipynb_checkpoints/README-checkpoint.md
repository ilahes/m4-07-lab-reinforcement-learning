![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Reinforcement Learning

## Overview

Reinforcement learning (RL) takes a fundamentally different approach from supervised and unsupervised learning: instead of learning from a static dataset, an agent learns by *interacting* with an environment and maximizing cumulative reward over time. This trial-and-error loop — observe state, take action, receive reward, update strategy — is how RL agents master games, control robots, and optimize complex systems.

In this lab you will implement tabular RL algorithms from scratch using Gymnasium environments. You'll start by exploring two classic grid-world problems — FrozenLake and Taxi — then build a Q-Learning agent, apply it to both environments, and finally implement SARSA to compare on-policy and off-policy learning.

By coding these algorithms yourself (rather than calling a library function), you'll develop intuition for how value tables converge, why exploration-exploitation trade-offs matter, and where the theoretical differences between Q-Learning and SARSA show up in practice.

## Learning Goals

By the end of this lab, you should be able to:

- Interact programmatically with Gymnasium environments (observation/action spaces, step loop, episode resets).
- Implement Q-Learning from scratch and tune its hyperparameters.
- Train and evaluate a tabular RL agent on discrete-state environments.
- Compare Q-Learning (off-policy) and SARSA (on-policy) and explain their behavioral differences.

## Setup and Context

You'll work inside a Jupyter Notebook for this lab. All analysis, code, and written interpretations should live in a single notebook so that your reasoning is visible alongside the output.

Gymnasium (the maintained successor to OpenAI Gym) provides standardized RL environments with consistent APIs. FrozenLake is a 4×4 grid where the agent must reach a goal tile without falling into holes, while Taxi is a 5×5 grid where a taxi must pick up and drop off a passenger at the correct location.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install numpy pandas matplotlib gymnasium
```

## Getting Started

1. Create a new Jupyter Notebook called **`m4-07-reinforcement-learning.ipynb`**.
2. In the first cell, import everything you'll need:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
```

3. Work through the tasks in order. Each task builds on the previous one.
4. Include markdown cells between code cells to explain your observations.

## Tasks

### Task 1: Environment Exploration

1. Create the **FrozenLake-v1** environment (`gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="ansi")`).
2. Print the observation space and action space. Explain what each action index (0–3) means.
3. Run **5 episodes** with random actions. For each episode, print the total reward and the number of steps taken.
4. Render and display the FrozenLake grid (the `ansi` render mode returns a string you can print).
5. Repeat steps 1–3 for the **Taxi-v3** environment (`gym.make("Taxi-v3")`). Print observation and action spaces and run 5 random-agent episodes.
6. In a markdown cell, compare the two environments: How large is each state space? How many actions are available? Why is Taxi harder than FrozenLake?

### Task 2: Q-Learning on FrozenLake

1. Initialize a Q-table of zeros with shape `(n_states, n_actions)` for FrozenLake.
2. Set hyperparameters:
   - `alpha = 0.8` (learning rate)
   - `gamma = 0.95` (discount factor)
   - `epsilon = 1.0` (initial exploration rate)
   - `epsilon_decay = 0.995`
   - `min_epsilon = 0.01`
3. Implement the Q-Learning update rule:
   ```
   Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
   ```
4. Train for **10,000 episodes**. After each episode, decay epsilon. Record the reward for every episode.
5. Plot the **cumulative reward** over episodes using a rolling average (window of 100) to smooth the curve.
6. Print the final Q-table. In a markdown cell, describe what the agent learned: for the start state, which action has the highest Q-value? Does the learned policy make intuitive sense given the grid layout?

### Task 3: Q-Learning on Taxi

1. Apply your Q-Learning implementation to **Taxi-v3** (reset the Q-table for the new environment's state/action dimensions).
2. Use the same hyperparameters from Task 2.
3. Train for **20,000 episodes**. Record the reward per episode.
4. Plot the **average reward per 100 episodes** over the training run.
5. Evaluate the trained agent on **100 test episodes** (set `epsilon = 0` for pure exploitation). Report the **average reward** and **success rate** (episodes where total reward > 0).
6. In a markdown cell, discuss: How does the training curve compare to FrozenLake? How many episodes did the agent need before performance stabilized?

### Task 4: SARSA Comparison

1. Implement the **SARSA** update rule:
   ```
   Q(s, a) ← Q(s, a) + α [ r + γ · Q(s', a') − Q(s, a) ]
   ```
   The key difference: SARSA uses the *actual next action* `a'` (chosen by the policy), not the greedy `max` action.
2. Train SARSA on **Taxi-v3** for **20,000 episodes** with the same hyperparameters.
3. Plot both learning curves (Q-Learning and SARSA) on the **same figure** — average reward per 100 episodes for both.
4. Evaluate both agents on **100 test episodes** each. Report average reward and success rate for both.
5. In a markdown cell, answer:
   - Which algorithm converged faster?
   - Which achieved a higher final reward?
   - Explain the fundamental difference between on-policy (SARSA) and off-policy (Q-Learning). In what situations might you prefer one over the other?

## Submission

### What to submit
- `m4-07-reinforcement-learning.ipynb` — completed notebook with all four tasks.

### Definition of done (checklist)
- [ ] Both environments are explored with random agents; observation/action spaces are printed.
- [ ] Q-Learning is implemented from scratch and trained on FrozenLake with a plotted learning curve.
- [ ] Q-Learning is applied to Taxi-v3 with evaluation metrics reported.
- [ ] SARSA is implemented and compared with Q-Learning on the same figure.
- [ ] Every task includes at least one markdown cell with interpretation.
- [ ] The notebook runs top-to-bottom without errors (`Kernel → Restart & Run All`).

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete reinforcement learning"
git push origin main
```

Then open a **Pull Request** on the original repository with a brief description of your work.
