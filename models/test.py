import config
from env.wumpus_world_env import WumpusWorldEnv
from models.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def save_steps_plot(steps_to_win, agent_name):
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
    plt.savefig(f"steps_to_win_distribution_{agent_name}.png")
    plt.close()


if __name__ == "__main__":
    agents = [
        {"name": "basic", "path": "random_map_weights/model_final_3pit.pt", "sensation_maps": False},
        {"name": "greedy", "path": "greedy_agent_weights/model_final.pt", "sensation_maps": False},
        {"name": "sensation", "path": "sensation_agent_weights/model_final_1-2-3pit.pt", "sensation_maps": True},
    ]

    results = [{"agent": agent["name"], "won": 0, "dead": 0, "steps": []} for agent in agents]

    base_env = WumpusWorldEnv(grid_size=4, default_map=False, num_of_pits=3, sensation_maps=False)

    done = False
    agent_instances = []
    env_instances = []

    for i, agent in enumerate(agents):
        env_instances.append(WumpusWorldEnv(grid_size=4, default_map=False, num_of_pits=3, sensation_maps=agent["sensation_maps"]))
        state_dim = env_instances[i].observation_space.shape[0]
        action_dim = env_instances[i].action_space.n
        agent_instances.append(DQNAgent(state_dim, action_dim, epsilon=0, epsilon2=0, min_epsilon=0, min_epsilon2=0))
        agent_instances[i].load_model(f"{agent['path']}")

    max_episodes = 5000
    max_steps = 100
    won_games = 0
    dead_games = 0
    not_possible_games = 0
    steps_to_win = []

    episode = 0
    print("Starting test...")

    while episode < max_episodes:
        obs, info = base_env.reset()
        map_info = base_env.get_map_info()
        possible_to_win = info["possible_to_win"]
        if not possible_to_win:
            not_possible_games += 1
            continue

        for i, agent in enumerate(agent_instances):
            env_instances[i].reset()
            obs, info = env_instances[i].set_map_info(map_info)
            done = False
            steps = 0

            while not done and steps < max_steps:
                action = agent.act(obs)
                if action is None:
                    break

                obs, reward, done, truncated, info = env_instances[i].step(action)
                if info["dead"]:
                    results[i]["dead"] += 1
                    break
                if info["won"]:
                    results[i]["won"] += 1
                    results[i]["steps"].append(steps)
                    break
                steps += 1

        episode += 1

    for result in results:
        steps = result["steps"]
        if steps:
            save_steps_plot(steps, result["agent"])
        print(f"{result["agent"]}:")
        print(f"  Wins: {result['won']} / {max_episodes}")
        print(f"  Deaths: {result['dead']}")
        print(f"  Win rate: {(result['won'] / max_episodes):.2%}")
        print(f"  Survival rate: {(1 - result['dead'] / max_episodes):.2%}")
        print()
    print(f"Not possible to win: {not_possible_games} / {max_episodes}")
