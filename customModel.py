import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_features):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(num_features, 10)
        self.act1 = nn.Sigmoid()
        self.hidden2 = nn.Linear(10, 8)
        self.act2 = nn.Sigmoid()
        self.hidden3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x)
        return x