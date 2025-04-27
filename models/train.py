import time
import pandas as pd
from env.wumpus_world_env import WumpusWorldEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import os

def save_plots(episode_rewards, episode_losses, episode, dir, window=1000):
    """ Save reward and loss plots as images with rolling average. """
    plt.figure(figsize=(24, 8))

    # Calculate rolling averages
    rewards_smoothed = pd.Series(episode_rewards).rolling(window, min_periods=1).mean()
    losses_smoothed = pd.Series(episode_losses).rolling(window, min_periods=1).mean()

    plt.subplot(1, 2, 1)
    plt.plot(rewards_smoothed, label='Episode Reward (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode (Smoothed)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(losses_smoothed, label='Average Loss per Episode (Smoothed)', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Episode (Smoothed)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{dir}/training_progress_episode_{episode}.png")
    plt.close()


if __name__ == "__main__":
    env = WumpusWorldEnv(grid_size=4, default_map=True)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    episodes = 20_000
    max_steps = 100
    target_update = 30
    checkpoint_interval = 2_000
    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    start_checkpoint = time.time()
    start = start_checkpoint

    reward_history = []
    loss_history = []

    epsilon = 0.38
    epsilon2 = 0.5
    epsilon_decay = 7e-5
    epsilon_decay2 = 5e-4
    min_epsilon = 5e-3

    used_epsilon_2 = False

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        if used_epsilon_2: # If we used epsilon2 in the last episode, we pull the values of epsilon2
            epsilon2 = agent.epsilon

        used_epsilon_2 = False # Reset the flag for the next episode

        # Set the epsilon and epsilon_decay as it starts in the stage 1 of the epsilon greedy strategy (gold not taken)
        agent.epsilon = epsilon
        agent.epsilon_decay = epsilon_decay

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            tookGold = info["tookGold"] # If the agent took gold in this step we start stage 2 of epsilon greedy strategy
            if tookGold:
                used_epsilon_2 = True # Set the flag to True to indicate we used epsilon2
                epsilon = agent.epsilon

                agent.epsilon = epsilon2
                agent.epsilon_decay = epsilon_decay2
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
            print(f"{checkpoint_interval} episodes took {((time.time() - start_checkpoint)/60):.2f} minutes.")
            start_checkpoint = time.time()
            print(f"Checkpoint model saved to {path}")
            print(f"Episode {episode}/{episodes}: Reward = {total_reward}, Epsilon1 = {epsilon:.3f}, Epsilon2 = {epsilon2:.3f}, Loss = {loss_history[-1]:.4f}")
            save_plots(reward_history, loss_history, episode, model_dir)

    print(f"Training completed in {((time.time() - start_checkpoint)/60):.2f} minutes.")
    path = os.path.join(model_dir, "model_final.pt")
    agent.save(path)
    print(f"Final model saved to {path}")
    save_plots(reward_history, loss_history, "final", model_dir)
    env.close()
