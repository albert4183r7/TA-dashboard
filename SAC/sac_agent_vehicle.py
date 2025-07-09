import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pickle
import os
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.FloatTensor([e[0] for e in experiences])
        actions = torch.FloatTensor([e[1] for e in experiences])
        rewards = torch.FloatTensor([e[2] for e in experiences]).unsqueeze(1)
        next_states = torch.FloatTensor([e[3] for e in experiences])
        dones = torch.FloatTensor([e[4] for e in experiences]).unsqueeze(1)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)  # This allows len(replay_buffer)

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean_linear(x)) 
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        action = mean + std * noise
        log_prob = -0.5 * ((noise ** 2) + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=1, keepdim=True)
        action = torch.tanh(action) 
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)
        return action, log_prob

class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, target_entropy=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared Policy and Q Networks
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.q_net1 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.q_net2 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr)

        # Target Networks
        self.target_q_net1 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net2 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -action_dim if target_entropy is None else target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Per-vehicle replay buffers
        self.replay_buffers = {}  # { veh_id: ReplayBuffer }
        self.default_buffer_size = 100000
        self.batch_size = 256

        # Loss tracking
        self.policy_loss = 0
        self.q1_loss = 0
        self.q2_loss = 0
        self.alpha_loss = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy_net.sample(state)
        return action.cpu().numpy()[0]

    def add_experience(self, veh_id, state, action, reward, next_state, done):
        """Store experience for a given vehicle"""
        if veh_id not in self.replay_buffers:
            self.replay_buffers[veh_id] = ReplayBuffer(self.default_buffer_size, self.batch_size)
        self.replay_buffers[veh_id].add(state, action, reward, next_state, done)

    def has_enough_data(self, veh_id):
        return len(self.replay_buffers.get(veh_id, [])) >= self.batch_size

    def train_for_vehicle(self, veh_id):
        """Train using data from a specific vehicle"""
        if not self.has_enough_data(veh_id):
            return

        buffer = self.replay_buffers[veh_id]
        states, actions, rewards, next_states, dones = buffer.sample()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q network update
        q1 = self.q_net1(states, actions)
        q2 = self.q_net2(states, actions)
        next_actions, next_log_probs = self.policy_net.sample(next_states)
        next_q1 = self.target_q_net1(next_states, next_actions)
        next_q2 = self.target_q_net2(next_states, next_actions)
        q_target = rewards + (1 - dones) * self.gamma * (torch.min(next_q1, next_q2) - self.alpha * next_log_probs)

        q1_loss = F.mse_loss(q1, q_target.detach())
        q2_loss = F.mse_loss(q2, q_target.detach())

        self.q_optimizer1.zero_grad()
        q1_loss.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        self.q_optimizer2.step()

        # Policy network update
        new_actions, log_probs = self.policy_net.sample(states)
        q1_pi = self.q_net1(states, new_actions)
        q2_pi = self.q_net2(states, new_actions)

        policy_loss = (self.alpha * log_probs - torch.min(q1_pi, q2_pi)).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update losses for logging
        self.policy_loss = policy_loss.item()
        self.q1_loss = q1_loss.item()
        self.q2_loss = q2_loss.item()
        self.alpha_loss = alpha_loss.item()

        # Soft update target networks
        self._soft_update(self.q_net1, self.target_q_net1)
        self._soft_update(self.q_net2, self.target_q_net2)

    def _soft_update(self, local, target):
        for t_param, l_param in zip(target.parameters(), local.parameters()):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)

    def save_all_buffers(self, base_path='SAC/model/replay_buffer_{}.pkl'):
        for veh_id, buffer in self.replay_buffers.items():
            path = base_path.format(veh_id)
            with open(path, 'wb') as f:
                pickle.dump(list(buffer.memory), f)

    def load_all_buffers(self, base_path='SAC/model/replay_buffer_{}.pkl'):
        for veh_id in self.replay_buffers:
            path = base_path.format(veh_id)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.replay_buffers[veh_id].memory = deque(data, maxlen=self.default_buffer_size)