import torch
import torch.nn as nn

class ChessCNN(nn.Module):
    def __init__(self, number_of_legal_moves):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjusted input size for the fully connected layer
        self.fc2 = nn.Linear(128, number_of_legal_moves)  # Output layer
        self.relu = nn.ReLU()
        self.log_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        return torch.softmax(self.fc2(x), dim=1)  # Output probabilities for each move
