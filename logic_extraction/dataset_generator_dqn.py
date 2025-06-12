import csv
import pygame

import config
from env.wumpus_world_env import WumpusWorldEnv
from models.dqn_agent import DQNAgent

env = WumpusWorldEnv(grid_size=4, default_map=False, num_of_pits=3, sensation_maps=True)

obs, _ = env.reset()
done = False

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, epsilon=0, epsilon2=0, min_epsilon=0, min_epsilon2=0)  # No exploration

agent.load_model("../models/sensation_agent_weights/model_final_1-2-3pit.pt")

dataset = []
max_episodes = 5000
max_steps = 100

episode = 0
print("Starting data collection...")

while episode < max_episodes:
    obs, info = env.reset()
    possible_to_win = info["possible_to_win"]
    if not possible_to_win:
        continue
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = agent.act(obs)
        if action is None:
            break


        # mapping [stench, breeze, glitter, bump, scream, hasgold, on_entrance]
        input_features = obs[:7].tolist() if hasattr(obs, "tolist") else list(obs[:7])
        action_for_dataset = action
        if action > 1: # Turn left or right is action 1 and 2 so we need to subtract 1 to map it to just 1
            action_for_dataset = action - 1
        dataset.append(input_features + [action_for_dataset])

        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    episode += 1

# Save the dataset to a CSV file
output_file = "datasets/sensation_agent_1-2-3pit_dataset.csv"
header = ["stench", "breeze", "glitter", "bump", "scream", "hasgold", "on_entrance", "action"]

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(dataset)

print(f"Dataset saved to {output_file} with {len(dataset)} samples.")
pygame.quit()
