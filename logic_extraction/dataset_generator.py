import csv
import pygame

from env.wumpus_world_env import WumpusWorldEnv
from models.dqn_agent import DQNAgent

env = WumpusWorldEnv(grid_size=4, default_map=True, num_of_pits=3)

obs, _ = env.reset()
done = False

state_dim = 10
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, epsilon=0, epsilon2=0)  # No exploration

agent.load_model("../models/default_map_weights/model_final_3pit.pt")

dataset = []
max_episodes = 500
max_steps = 80

episode = 0
print("Starting data collection...")

while episode < max_episodes:
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = agent.act(obs)
        if action is None:
            break

        # mapping [stench, breeze, glitter, bump, scream]
        input_features = obs[:5].tolist() if hasattr(obs, "tolist") else list(obs[:5])
        dataset.append(input_features + [action])

        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    episode += 1

# Save the dataset to a CSV file
output_file = "dataset.csv"
header = ["stench", "breeze", "glitter", "bump", "scream", "action"]

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(dataset)

print(f"Dataset saved to {output_file} with {len(dataset)} samples.")
pygame.quit()
