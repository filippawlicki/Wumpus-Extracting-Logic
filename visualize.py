import sys

import config
from env.wumpus_world_env import WumpusWorldEnv
from env.render import Renderer
import time
import pygame
from models.dqn_agent import DQNAgent

env = WumpusWorldEnv(grid_size=4, default_map=False, num_of_pits=3)

obs, _ = env.reset()
done = False

state_dim = config.STATE_DIM
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, epsilon=0, epsilon2=0, min_epsilon=0, min_epsilon2=0) # No exploration

agent.load("models/sensation_agent_weights/model_ep20000.pt")

max_steps = 100
steps = 0

while True:
    obs, info = env.reset()
    done = False
    steps = 0

    possible_to_win = info["possible_to_win"]
    if not possible_to_win:
        print("This map is not possible to win.")
    while not done:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        action = agent.act(obs)
        if action is not None:
            obs, reward, done, truncated, info = env.step(action)
            print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")
            time.sleep(0.2)

            steps += 1
            if steps >= max_steps:
                done = True
                break


pygame.quit()
