import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv2D Layer
class Conv2D(nn.Module):
    def __init__(self, input_shape, kernel_size, depth):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_shape[2], out_channels=depth, kernel_size=kernel_size)

    def forward(self, x):
        # Expecting input in shape (batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)  # Reorder from (h, w, c) to (c, h, w) for PyTorch
        return F.relu(self.conv(x)).permute(0, 2, 3, 1)  # Change back to (h, w, c)

# MaxPool2D Layer
class MaxPool2D(nn.Module):
    def __init__(self, pool_size):
        super(MaxPool2D, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Reorder for PyTorch
        return self.pool(x).permute(0, 2, 3, 1)  # Change back to (h, w, c)

# Flatten Layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Dense Layer (Fully Connected)
class Dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(Dense, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)