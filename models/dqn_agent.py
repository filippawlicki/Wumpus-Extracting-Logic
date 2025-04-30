import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.9, lr=1e-3, epsilon=0.9, epsilon_decay=3e-4, min_epsilon=5e-3,
                 epsilon2=0.9, epsilon_decay2=3e-4, min_epsilon2=5e-3):
        self._last_loss = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.q_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(50000)
        self.batch_size = 128
        self.gamma = gamma

        self.tookGold = False

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon2 = epsilon2
        self.epsilon_decay2 = epsilon_decay2
        self.min_epsilon = min_epsilon
        self.min_epsilon2 = min_epsilon2

        self.action_dim = action_dim
        self.state_dim = state_dim


    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        if self.tookGold: # Exploration phase
            if random.random() < self.epsilon2:
                return random.randint(0, self.action_dim - 1)
        else:
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state).cpu().numpy()
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.from_numpy(states).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)


        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, expected_q_values)
        self._last_loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.target_net.parameters(), 5.0)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
        if self.tookGold:
            self.epsilon2 = max(self.min_epsilon2, self.epsilon2 - self.epsilon_decay2)


    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.q_net.eval()

    def get_last_loss(self):
        return self._last_loss if self._last_loss is not None else 0.0

    def load_epsilon(self, num_of_pits):
        if num_of_pits == 1:
            self.epsilon = 1.0
            self.epsilon2 = 0.7
            self.min_epsilon = 0.01
            self.min_epsilon2 = 0.01
            self.epsilon_decay = 2e-4
            self.epsilon_decay2 = 4e-4
        elif num_of_pits == 2:
            self.epsilon = 0.02
            self.epsilon2 = 0.01
            self.min_epsilon = 0.02
            self.min_epsilon2 = 0.01
            self.epsilon_decay = 0
            self.epsilon_decay2 = 0
        elif num_of_pits == 3:
            self.epsilon = 0.38
            self.epsilon2 = 0.5
            self.min_epsilon = 5e-3
            self.min_epsilon2 = 0.005
            self.epsilon_decay = 7e-5
            self.epsilon_decay2 = 5e-4



