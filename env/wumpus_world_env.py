import gymnasium as gym
import numpy as np
import random
from env.render import Renderer
import config



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
        self.observation_space = gym.spaces.MultiDiscrete(10) # [Stench, Breeze, Glitter, Bump, Scream, hasgold, orientation]
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.visited.fill(False)
        self.num_of_pits = num_of_pits
        self.steps_taken = 0
        # Top left corner 2x2 box is prohibited for creating the map as its the entrance
        self.prohibited_box = [(0, 0), (0, 1), (1, 0), (1, 1)]

        self.renderer = Renderer(self)
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.
        """
        self.wumpus_alive = True
        self.agent_has_gold = False
        self.has_arrow = True
        self.steps_taken = 0
        self.bump = False
        self.scream = False

        self.grid = np.zeros((self.grid_size, self.grid_size, 2), dtype=int) # breeze, stench, glitter
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.visited.fill(False)


        if self.default_map:
            self.entrance = (0, 0)
            self.agent_dir = 2  # 0: North, 1: East, 2: South, 3: West
            self.wumpus_pos = (0, 2)
            self.gold_pos = (1, 2)
            self.pit_pos = [(2, 0)]
            if self.num_of_pits > 1:
                self.pit_pos.append((2, 2))
            if self.num_of_pits > 2:
                self.pit_pos.append((3, 3))
        else:
            self.entrance = (0, 0)
            self.agent_dir = random.randint(0, 3)
            while True:
                self.wumpus_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                if self.wumpus_pos != self.entrance and self.wumpus_pos not in self.prohibited_box:
                    break

            while True:
                self.gold_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                if self.gold_pos != self.entrance and self.gold_pos != self.wumpus_pos and self.gold_pos not in self.prohibited_box:
                    break

            self.pit_pos = []
            while len(self.pit_pos) < self.num_of_pits: # Randomly place pits
                x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
                if (x, y) != self.entrance and (x, y) != self.wumpus_pos and (x, y) != self.gold_pos and (x, y) not in self.pit_pos and (x, y) not in self.prohibited_box:
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
            hasgold = 1
        else:
            hasgold = 0
            glitter = 1 if self.agent_pos == self.gold_pos else 0
        orientation = self.agent_dir
        if self.agent_pos == self.entrance:
            entrance = 1
        else:
            entrance = 0
        bump = self.bump
        scream = self.scream
        self.bump = False
        self.scream = False
        posx = self.agent_pos[0]
        posy = self.agent_pos[1]

        return np.array([stench, breeze, glitter, bump, scream, hasgold, entrance, orientation, posx, posy], dtype=np.float32)


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

        reward = -1 # Default reward for each step
        done = False
        x, y = self.agent_pos
        new_x, new_y = x, y
        tookGold = False
        won = False
        log = False
        self.scream = False

        if action == config.ACTION_MOVE_FORWARD:
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.agent_dir]
            new_x = x + dx
            new_y = y + dy
            if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size): # Bump into wall
                self.bump = True
                if log:
                    print("Bump into wall: -5 reward")
                reward = -5
            else:
                if self.visited[new_x, new_y]:
                    reward = -0.001 # Small penalty for revisiting a cell
                else:
                    self.visited[new_x, new_y] = True
                    reward = 50 # Small reward for visiting a new cell
                self.agent_pos = new_x, new_y


        elif action == config.ACTION_TURN_LEFT:
            self.agent_dir = (self.agent_dir - 1) % 4
            #reward = -5

        elif action == config.ACTION_TURN_RIGHT:
            self.agent_dir = (self.agent_dir + 1) % 4
            #reward = -5

        elif action == config.ACTION_GRAB:
            if self.agent_pos == self.gold_pos and not self.agent_has_gold: # Grab gold
                self.agent_has_gold = True
                tookGold = True
                if log:
                    print("Grab gold: +500 reward")
                reward = 500
            else: # Tried to grab without being on gold
                if log:
                    print("Tried to grab without being on gold: -20 reward")
                reward = -20


        elif action == config.ACTION_SHOOT:
            if self.has_arrow:
                self.has_arrow = False
                if self._shoot(): # Shoot Wumpus
                    if log:
                        print("Shoot Wumpus: +300 reward")
                    self.scream = True
                    reward = 300
            else: # Tried to shoot without an arrow
                if log:
                    print("Tried to shoot without an arrow: -20 reward")
                reward = -20

        elif action == config.ACTION_CLIMB:
            if self.agent_pos == self.entrance and self.agent_has_gold: # Exit with gold
                if log:
                    print("Exit with gold: +1000 reward")
                reward = 1000
                won = True
                done = True
            elif self.agent_pos != self.entrance or not self.agent_has_gold: # Tried to climb without being at the entrance or without gold
                if log:
                    print("Tried to climb without being at the entrance: -20 reward")
                reward = -20


        x, y = self.agent_pos
        if self.wumpus_alive and (x, y) == self.wumpus_pos: # Death by Wumpus
            if log:
                print("Death by Wumpus: -1000 reward")
            reward = -1000
            done = True
        elif (x, y) in self.pit_pos: # Death by pit
            if log:
                print("Death by pit: -1000 reward")
            reward = -1000
            done = True

        self.steps_taken += 1
        if self.steps_taken >= 100: # Max steps
            if log:
                print("Max steps reached: -1000 reward")
            reward = -1000
            done = True

        return self._get_observation(), reward, done, False, {"tookGold": tookGold, "won": won}

    def render(self):
        self.renderer.render(self._get_observation())










