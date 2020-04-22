import random

import numpy as np
import torch
import torch.optim as optim
from network import QNetwork
from buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(object):
    def __init__(self, state_size, action_size, seed, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.seed = random.seed(seed)

        self.local_q_net = QNetwork(state_size, action_size, seed).to(device)
        self.target_q_net = QNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.local_q_net.parameters(), lr=config["LR"])

        self.memory = ReplayBuffer(action_size, config["BUFFER_SIZE"], config["BATCH_SIZE"], seed)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.config["UPDATE_EVERY"]

        if self.t_step == 0:
            # if agent experienced enough
            if len(self.memory) > self.config["BATCH_SIZE"]:
                experiences = self.memory.sample()
                # Learn from previous experiences
                self.learn(experiences, self.config["GAMMA"])

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_q_net.eval()
        with torch.no_grad():
            action_values = self.local_q_net(state)
        self.local_q_net.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        # Double Q Learning

        states, actions, rewards, next_states, dones = experiences

        # Get next action estimation with local q network
        q_targets_next_expected = self.local_q_net(next_states).detach()
        q_targets_next_expected_actions = q_targets_next_expected.max(1)[1].unsqueeze(1)

        # Calculate Next Targets
        q_targets_next = self.target_q_net(next_states).gather(1, q_targets_next_expected_actions)

        # Non over-estimated targets
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Expected value
        q_expected = self.local_q_net(states).gather(1, actions)

        loss = torch.nn.functional.mse_loss(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_q_net, self.target_q_net, self.config["TAU"])

    def soft_update(self, local_net, target_net, tau):
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
