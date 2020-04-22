import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layers=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        if layers is None:
            layers = [64, 64]
        self.seed = torch.manual_seed(seed)

        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=layers[0]),
            nn.ReLU()
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=layers[0], out_features=layers[1]),
            nn.ReLU()
        )

        self.action_layer = nn.Linear(in_features=layers[1], out_features=action_size)

    def forward(self, state):
        out = self.fc_1(state)
        out = self.fc_2(out)
        action = self.action_layer(out)
        return action
