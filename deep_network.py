import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    """ Xavier Weights intialization with 1/sqrt of hidden nodes. """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class DeepNetwork(nn.Module):
    """ Generic Neural network class for both Actor and Critic """
    def __init__(self, category_nn, action_size, state_size, hidden_units, seed, dropout=0.3):
        """Initialize Neural Network parameters and build model.
        Params
        ======
            category_nn (string) : Type of network (Actor/Critic)
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            dropout(float) : dropout value for the network nodes.
        """
        super(DeepNetwork, self).__init__()
        
        # seed for reproducibility
        torch.manual_seed(seed)

        # Type of network
        self.category_nn = category_nn

        # Droput value
        self.dropout = nn.Dropout(p=dropout)

        # Batch Normalization
        self.normalizer = nn.BatchNorm1d(state_size)

        # First Fully connected Layer.
        self.fc1 = nn.Linear(state_size, hidden_units[0])

        # Intialization of second and third fully connected layer.
        if self.category_nn == "Actor":
        	# second and third fully connected layer of actor model.
        	self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        	self.fc3 = nn.Linear(hidden_units[1], action_size)

        elif self.category_nn == "Critic":
            # second and third fully connected layer of critic model.
        	self.fc2 = nn.Linear(hidden_units[0]+action_size, hidden_units[1])
        	self.fc3 = nn.Linear(hidden_units[1], 1)

        else:
            raise TypeError("Only Actor and Critic Categories are allowed.")

        # Weight Intialization of all fully connected Layers
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        print("Intialization of Network for "+ self.category_nn)
        print("Layers of "+ self.category_nn)
        print(self.fc1)
        print(self.fc2)
        print(self.fc3)

    def forward(self,  states, actions=0):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        x = self.normalizer(states)

        if self.category_nn == "Actor":
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           x = F.tanh(self.fc3(x))

        elif self.category_nn == "Critic":
           x = F.relu(self.fc1(x))
           x = torch.cat((x, actions), dim=1)
           x = F.relu(self.fc2(x))
           x = self.dropout(x)
           x = self.fc3(x)

        return x
