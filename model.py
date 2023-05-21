import torch.nn as nn
import torch

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        results = torch.relu(self.fc1(x))
        results = torch.relu(self.fc2(results))
        results = self.fc3(results)
        return results
