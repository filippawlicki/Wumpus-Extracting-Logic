import sys

from env.wumpus_world_env import WumpusWorldEnv
from env.render import Renderer
import time
import pygame


def handle_input():
    """Reads key press using pygame."""
    keys = pygame.key.get_pressed()  # Get the current state of all keys

    if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:  # Quit the game
        pygame.quit()
        sys.exit()

    action = None
    if keys[pygame.K_w]:
        action = 0  # Forward
    elif keys[pygame.K_a]:
        action = 1  # Turn left
    elif keys[pygame.K_d]:
        action = 2  # Turn right
    elif keys[pygame.K_g]:
        action = 3  # Grab
    elif keys[pygame.K_SPACE]:
        action = 4  # Climb
    elif keys[pygame.K_RETURN]:
        action = 5 # Shoot
    return action

env = WumpusWorldEnv(default_map=True)

obs, _ = env.reset()
done = False

print("W - Forward, A - Turn left, D - Turn right, G - Grab, Enter - Shoot, Space - Climb, Q - Quit")

while not done:
    env.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break

    action = handle_input()
    if action is not None:
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")
        time.sleep(0.2)


pygame.quit()
