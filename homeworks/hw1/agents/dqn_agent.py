"""DQN Agent class."""

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy


import numpy as np
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim


class DQNNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[128, 128], dropout_p=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_sizes[1], action_size)
        )

    def forward(self, x):
        return self.layers(x)


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        lr=1e-3,
        buffer_size=50000,
        batch_size=64,
        target_update=500,
        learning_starts=0,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        hidden_sizes=[128, 128],
        dropout_p=0.2,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_update = target_update
        self.learning_starts = learning_starts
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer = deque(maxlen=buffer_size)
        self.learn_step = 0
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNet(state_size, action_size, hidden_sizes, dropout_p).to(self.device)
        self.target_net = DQNNet(state_size, action_size, hidden_sizes, dropout_p).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax(1).item()

    def step(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.learning_starts and len(self.buffer) >= self.batch_size:
            self.learn()

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones

    def learn(self):
        states, actions, rewards, next_states, dones = self.sample()
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon*self.epsilon_decay)
