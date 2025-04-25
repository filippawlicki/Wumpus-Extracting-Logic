import gymnasium as gym
from env.wumpus_world_env import WumpusWorldEnv
from dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import os

env = WumpusWorldEnv(grid_size=4, default_map=True)
state_dim = env.observation_space.n
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

episodes = 40000
max_steps = 50
target_update = 30
checkpoint_interval = 500
model_dir = "checkpoints"
os.makedirs(model_dir, exist_ok=True)

reward_history = []
loss_history = []

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay(done)
        state = next_state
        total_reward += reward
        if done:
            break

    reward_history.append(total_reward)
    loss_history.append(agent.get_last_loss())

    if episode % target_update == 0:
        agent.update_target()

    if episode % checkpoint_interval == 0 and episode > 0:
        path = os.path.join(model_dir, f"model_ep{episode}.pt")
        agent.save(path)
        print(f"✅ Saved model to {path}")
        print(f"Episode {episode}/{episodes}: Reward = {total_reward}, Epsilon = {agent.epsilon:.3f}, Loss = {loss_history[-1]:.4f}")

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(reward_history)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title("Loss per Episode")
plt.xlabel("Episode")
plt.ylabel("Loss")

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()
