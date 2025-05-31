import config
from env.wumpus_world_env import WumpusWorldEnv
from models.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def save_steps_plot(steps_to_win):
    """ Save a histogram showing the distribution of steps needed to win. """
    plt.figure(figsize=(12, 6))

    min_steps = min(steps_to_win)

    bins = np.arange(min_steps, max(steps_to_win) + 2) - 0.5
    plt.hist(steps_to_win, bins=bins, color='blue', alpha=0.7, rwidth=0.85)

    plt.xlabel('Number of Steps to Win')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Steps Needed to Win')
    plt.xticks(np.arange(min_steps, max(steps_to_win) + 1, step=1))
    plt.tight_layout()
    plt.savefig(f"steps_to_win_distribution.png")
    plt.close()


if __name__ == "__main__":
    env = WumpusWorldEnv(grid_size=4, default_map=False, num_of_pits=3)

    obs, _ = env.reset()
    done = False

    state_dim = config.STATE_DIM
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, epsilon=0, epsilon2=0, min_epsilon=0, min_epsilon2=0)  # No exploration
    agent.load_model("sensation_agent_weights/model_final_1pit.pt")

    max_episodes = 5000
    max_steps = 100
    won_games = 0
    dead_games = 0
    not_possible_games = 0
    steps_to_win = []

    episode = 0
    print("Starting test...")

    while episode < max_episodes:
        obs, info = env.reset()
        possible_to_win = info["possible_to_win"]
        if not possible_to_win:
            not_possible_games += 1
            continue
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.act(obs)
            if action is None:
                break

            obs, reward, done, truncated, info = env.step(action)
            if info["dead"]:
                dead_games += 1
                break
            if info["won"]:
                won_games += 1
                steps_to_win.append(steps)
                break
            steps += 1

        episode += 1

    save_steps_plot(steps_to_win)

    print(f"Test completed. Won {won_games}/{max_episodes} games. Win rate: {(won_games/max_episodes):.2%}")
    print(f"Survived {max_episodes - dead_games}/{max_episodes} games. Survival rate: {(max_episodes - dead_games)/max_episodes:.2%}")
    print(f"Not possible to win in {not_possible_games}/{max_episodes} games. Not possible rate: {(not_possible_games/max_episodes):.2%}")