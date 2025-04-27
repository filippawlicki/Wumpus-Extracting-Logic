import sys

from env.wumpus_world_env import WumpusWorldEnv
from env.render import Renderer
import time
import pygame
from models.dqn_agent import DQNAgent

env = WumpusWorldEnv(default_map=True)

obs, _ = env.reset()
done = False

state_dim = env.observation_space.n
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, epsilon=0)

agent.load("models/checkpoints/model_ep10000.pt")

max_steps = 100
steps = 0

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
