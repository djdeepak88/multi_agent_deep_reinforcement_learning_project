import random
import numpy as np
import torch
import torch.nn.functional as F
import copy
from collections import namedtuple, deque
from deep_network import DeepNetwork


class Agent:
    """Initialize DDPG Agent."""
    def __init__(self, seed, device, action_size, state_size, actor_hidden_units, actor_learning_rate, critic_hidden_units, critic_learning_rate, weight_decay, buffer_size, batch_size,tau):
        """ Intialize Hyperparameters for the Agent object.
            Local and target networks for Actor and Critic network
            Optimizers and update function for both network.
        """
        self.seed = seed
        self.device = device

        # Actor Network Hyperparameters Initialization
        self.actor_local = DeepNetwork("Actor",action_size, state_size, actor_hidden_units, seed).to(self.device)
        self.actor_target = DeepNetwork("Actor",action_size, state_size, actor_hidden_units, seed).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=actor_learning_rate)

        # Critic Network Hyperparameters Initialization
        self.critic_local = DeepNetwork("Critic",action_size, state_size, critic_hidden_units, seed).to(self.device)
        self.critic_target = DeepNetwork("Critic",action_size, state_size, critic_hidden_units, seed).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=critic_learning_rate,weight_decay=weight_decay)

        # Target Network Intialization #
        self.hard_update(self.critic_local, self.critic_target)
        self.hard_update(self.actor_local, self.actor_target)

        # OUN Noise function
        self.noise = OUNoise(action_size)

        # Replay Buffer memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, device)

        # Tau for soft update
        self.tau = tau

    def reset(self):
        """ Resetting the OUN noise """
        self.noise.reset()

    def act(self, states):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        actions += self.noise.noise()
        return np.clip(actions, -1, 1)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
    def hard_update(self,target, source):
        """
        Copy network parameters from source to target
        Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def load(self, actor_file, critic_file):
        self.actor_local.load_state_dict(torch.load(actor_file))
        self.critic_local.load_state_dict(torch.load(critic_file))
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of action space
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """ randomly sample the next state, action and rewards from the memory. """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    """Ornstein-Uhlenbeck process implementation"""
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        """ Intialize OUN class
        Params
        ======
            action_dimension (int): dimension of action space
            scale (float): scaling factor
            mu (float): mu value of normal distribution
            theta (float): theta value
            sigma (float): sigma value of normal distribution
        """
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        """Reset function"""
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        """Noise generation"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()
