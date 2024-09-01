import numpy as np
from utils import *
from models import Conv2D, MaxPool2D, Flatten, Dense
class SimpleCNN:
    def __init__(self):
        self.conv1 = Conv2D((28, 28, 1), 3, 8)  # 28x28x1 -> 26x26x8
        self.pool1 = MaxPool2D((2, 2))  # 26x26x8 -> 13x13x8
        self.flatten = Flatten()
        self.fc1 = Dense(13*13*8, 128)
        self.fc2 = Dense(128, 10)
        print("SimpleCNN initialized")
    
    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.pool1.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.fc2.forward(x)
        return softmax(x)
    
    def backward(self, y_pred, y_true, learning_rate):
        # Compute the gradient of the loss function
        d_output = y_pred - y_true
        d_output = self.fc2.backward(d_output, learning_rate)
        d_output = self.fc1.backward(d_output, learning_rate)
        d_output = self.flatten.backward(d_output)
        d_output = self.pool1.backward(d_output)
        d_output = self.conv1.backward(d_output, learning_rate)
