import gymnasium as gym
import numpy as np
import random
from env.render import Renderer

ACTION_MOVE_FORWARD = 0
ACTION_TURN_LEFT = 1
ACTION_TURN_RIGHT = 2
ACTION_GRAB = 3
ACTION_CLIMB = 4
ACTION_SHOOT = 5



class WumpusWorldEnv(gym.Env):
    """
    Wumpus World Environment for reinforcement learning.
    """

    def __init__(self, grid_size=4, default_map=True, num_of_pits=3):
        super(WumpusWorldEnv, self).__init__()
        self.bump = False
        self.scream = False
        self.pit_pos = None
        self.gold_pos = None
        self.wumpus_pos = None
        self.agent_dir = None
        self.entrance = None
        self.agent_has_gold = None
        self.has_arrow = None
        self.wumpus_alive = None
        self.agent_pos = None
        self.grid_size = grid_size
        self.grid = None
        self.default_map = default_map
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.MultiBinary(5) # [Stench, Breeze, Glitter, Bump, Scream]
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.num_of_pits = num_of_pits

        self.renderer = Renderer(self)
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.
        """
        self.wumpus_alive = True
        self.agent_has_gold = False
        self.has_arrow = True

        self.grid = np.zeros((self.grid_size, self.grid_size, 2), dtype=int) # breeze, stench, glitter


        if self.default_map:
            self.entrance = (0, 0)
            self.agent_dir = 2 # 0: North, 1: East, 2: South, 3: West
            self.wumpus_pos = (0, 2)
            self.gold_pos = (1, 2)
            self.pit_pos = [(2, 0)]
            if self.num_of_pits > 1:
                self.pit_pos.append((2, 2))
            if self.num_of_pits > 2:
                self.pit_pos.append((3, 3))
        else:
            self.entrance = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            self.agent_dir = random.randint(0, 3)
            while True:
                self.wumpus_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                if self.wumpus_pos != self.entrance:
                    break

            while True:
                self.gold_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                if self.gold_pos != self.entrance and self.gold_pos != self.wumpus_pos:
                    break

            self.pit_pos = []
            while len(self.pit_pos) < self.num_of_pits: # Randomly place pits
                x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
                if (x, y) != self.entrance and (x, y) != self.wumpus_pos and (x, y) != self.gold_pos and (x, y) not in self.pit_pos:
                    self.pit_pos.append((x, y))

        self.agent_pos = self.entrance
        self._update_perception()

        return self._get_observation(), {}



    def _update_perception(self):
        for x, y in self.pit_pos:
            for nx, ny in self._get_neighbors((x, y)):
                self.grid[nx, ny, 0] = 1 # Breeze

        if self.wumpus_alive:
            wx, wy = self.wumpus_pos
            for nx, ny in self._get_neighbors((wx, wy)):
                self.grid[nx, ny, 1] = 1 # Stench



    def _get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        if x > 0:
            neighbors.append((x-1, y))
        if x < self.grid_size - 1:
            neighbors.append((x+1, y))
        if y > 0:
            neighbors.append((x, y-1))
        if y < self.grid_size - 1:
            neighbors.append((x, y+1))
        return neighbors

    def _get_observation(self):
        x, y = self.agent_pos
        breeze = self.grid[x, y, 0]
        if self.wumpus_alive:
            stench = self.grid[x, y, 1]
        else:
            stench = 0
        if self.agent_has_gold:
            glitter = 0
        else:
            glitter = 1 if self.agent_pos == self.gold_pos else 0
        bump = self.bump
        scream = self.scream
        self.bump = False
        self.scream = False

        return np.array([stench, breeze, glitter, bump, scream], dtype=int)

    def _shoot(self):
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.agent_dir]
        x, y = self.agent_pos
        while 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            if (x, y) == self.wumpus_pos:
                self.wumpus_alive = False
                self.scream = True
                return True
            x += dx
            y += dy
        return False

    def step(self, action):
        """
        Takes a step in the environment based on the action.
        """

        reward = 0 # Default reward for each step
        done = False
        x, y = self.agent_pos
        new_x, new_y = x, y
        tookGold = False

        if action == ACTION_MOVE_FORWARD:
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.agent_dir]
            new_x = x + dx
            new_y = y + dy
            if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size): # Bump into wall
                self.bump = True
                reward += -5
            else:
                self.agent_pos = new_x, new_y
                if self.visited[new_x, new_y]:  # If the agent has already visited this cell
                    reward += -0.001
                else:  # If the agent has not visited this cell
                    reward += 50
                    self.visited[new_x, new_y] = True



        elif action == ACTION_TURN_LEFT:
            self.agent_dir = (self.agent_dir - 1) % 4
            reward += -5

        elif action == ACTION_TURN_RIGHT:
            self.agent_dir = (self.agent_dir + 1) % 4
            reward += -5

        elif action == ACTION_GRAB:
            if self.agent_pos == self.gold_pos and not self.agent_has_gold: # Grab gold
                self.agent_has_gold = True
                tookGold = True
                reward += 500
            elif self.agent_pos != self.gold_pos: # Tried to grab without being on gold
                reward += -20


        elif action == ACTION_SHOOT:
            if self.has_arrow:
                self.has_arrow = False
                if self._shoot(): # Shoot Wumpus
                    reward += 300
            else: # Tried to shoot without an arrow
                reward += -20

        elif action == ACTION_CLIMB:
            if self.agent_pos == self.entrance and self.agent_has_gold: # Exit with gold
                reward += 1000
                done = True
            elif self.agent_pos != self.entrance: # Tried to climb without being at the entrance
                reward += -20


        x, y = self.agent_pos
        if self.wumpus_alive and (x, y) == self.wumpus_pos: # Death by Wumpus
            reward += -1000
            done = True
        elif (x, y) in self.pit_pos: # Death by pit
            reward += -1000
            done = True

        return self._get_observation(), reward, done, False, {"tookGold": tookGold}

    def render(self):
        self.renderer.render(self._get_observation())










