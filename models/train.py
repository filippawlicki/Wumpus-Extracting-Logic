import time

import numpy as np
import pandas as pd
from env.wumpus_world_env import WumpusWorldEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import os

def save_plots(episode_rewards, episode_losses, episode, dir, window=25):
    """ Save reward and loss plots as images with rolling average. """
    plt.figure(figsize=(24, 8))

    # Calculate rolling averages
    rewards_smoothed = pd.Series(episode_rewards).rolling(window, min_periods=1).mean()
    losses_smoothed = pd.Series(episode_losses).rolling(window, min_periods=1).mean()

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward (Original)', alpha=0.3, linewidth=0.25, color='blue')
    plt.plot(rewards_smoothed, label='Episode Reward (Smoothed)', linewidth=1, color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode (Smoothed)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episode_losses, label='Average Loss (Original)', alpha=0.3, linewidth=0.5, color='red')
    plt.plot(losses_smoothed, label='Average Loss (Smoothed)', linewidth=2, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Episode (Smoothed)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{dir}/training_progress_episode_{episode}.png")
    plt.close()


if __name__ == "__main__":
    env = WumpusWorldEnv(grid_size=4, default_map=False, num_of_pits=1)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    agent.load_epsilon(env.num_of_pits) # Load epsilon values based on the number of pits

    episodes = 20_000
    max_steps = 100
    checkpoint_interval = 2_000
    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    start_checkpoint = time.time()
    start = start_checkpoint

    reward_history = []
    loss_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        agent.tookGold = False # Reset tookGold flag at the start of each episode

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            tookGold = info["tookGold"] # If the agent took gold in this step we start stage 2 of epsilon greedy strategy
            if tookGold:
                agent.tookGold = True
            agent.remember(state, action, reward, next_state, done)
            agent.replay(done)
            agent.update_target() # Soft update of the target network
            state = next_state
            total_reward += reward
            if done:
                break

        reward_history.append(total_reward)
        loss_history.append(agent.get_last_loss())

        if episode % checkpoint_interval == 0 and episode > 0:
            path = os.path.join(model_dir, f"model_ep{episode}.pt")
            agent.save(path)
            print(f"{checkpoint_interval} episodes took {((time.time() - start_checkpoint)/60):.2f} minutes.")
            start_checkpoint = time.time()
            print(f"Checkpoint model saved to {path}")
            window = 25
            mean_reward = np.mean(reward_history[-window:])
            mean_loss = np.mean(loss_history[-window:])
            print(
                f"{'='*50}\n"
                f"Episode {episode}/{episodes}:\n"
                f" (loss and reward are a mean from the last {window} episodes)\n"
                f"Reward = {mean_reward:.3f},\n"
                f"Epsilon1 = {agent.epsilon:.3f}, Epsilon2 = {agent.epsilon2:.3f},\n"
                f"Loss = {mean_loss:.3f}\n"
                f"{'=' * 50}\n"
            )
            save_plots(reward_history, loss_history, episode, model_dir)

    print(f"Training completed in {((time.time() - start_checkpoint)/60):.2f} minutes.")
    path = os.path.join(model_dir, "model_final.pt")
    agent.save(path)
    print(f"Final model saved to {path}")
    save_plots(reward_history, loss_history, "final", model_dir)
    env.close()
