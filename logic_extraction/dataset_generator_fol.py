import csv
import pygame

from env.wumpus_world_env import WumpusWorldEnv
from first_order_logic.fol_agent import FOLAgent

env = WumpusWorldEnv(grid_size=4, default_map=False, num_of_pits=3)

obs, _ = env.reset()
done = False


dataset = []
max_episodes = 5000
max_steps = 100

agent = FOLAgent(env, rendering=False, log=False)

episode = 0
print("Starting data collection...")

while episode < max_episodes:
    # obs, _ = env.reset()
    # agent = FOLAgent(env, testing=False, log=False)
    obs = agent.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        action, has_won, next_obs = agent.act()

        # mapping [stench, breeze, glitter, bump, scream, hasgold, on_entrance]
        input_features = obs[:7].tolist() if hasattr(obs, "tolist") else list(obs[:7])
        action_for_dataset = action
        if action > 1: # Turn left or right is action 1 and 2 so we need to subtract 1 to map it to just 1
            action_for_dataset = action - 1
        dataset.append(input_features + [action_for_dataset])

        if has_won:
            break
        obs = next_obs
        steps += 1

    episode += 1

# Save the dataset to a CSV file
output_file = "datasets/fol_3pit_random_map_dataset.csv"
header = ["stench", "breeze", "glitter", "bump", "scream", "hasgold", "on_entrance", "action"]

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(dataset)

print(f"Dataset saved to {output_file} with {len(dataset)} samples.")
pygame.quit()
