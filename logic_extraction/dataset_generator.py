import csv
import pygame

import config
from env.wumpus_world_env import WumpusWorldEnv
from first_order_logic.fol_agent import FOLAgent
from models.dqn_agent import DQNAgent

max_episodes = 5_000
max_steps = 100
grid_size = 7
num_of_pits = 3
buffer_size = 10
# agent_name = "fol"
agent_name = "sensation"
sensation_maps = True if agent_name == "sensation" else False

env = WumpusWorldEnv(grid_size=grid_size, default_map=False, num_of_pits=num_of_pits, sensation_maps=sensation_maps, buffer_size=buffer_size, max_steps=max_steps)

obs, _ = env.reset()
done = False

if agent_name == "fol":
    agent = FOLAgent(env, rendering=False, log=False)
else:
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, epsilon=0, epsilon2=0, min_epsilon=0, min_epsilon2=0)  # No exploration

    agent.load_model(f"../models/sensation_agent_weights/bigger_grid_size/model_final_{num_of_pits}pit_size_4.pt")


dataset = []

episode = 0
print("Starting data collection...")

while episode < max_episodes:
    if episode % 100 == 0:
        print(f"Episode {episode}/{max_episodes}")
    obs, info = agent.reset() if agent_name == "fol" else env.reset()
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
output_file = f"datasets/{agent_name}_agent_{num_of_pits}pit_size_{grid_size}_dataset.csv"
header = ["stench", "breeze", "glitter", "bump", "scream", "hasgold", "on_entrance", "action"]

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(dataset)

print(f"Dataset saved to {output_file} with {len(dataset)} samples.")
pygame.quit()
