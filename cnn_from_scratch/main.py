from train import *
import numpy as np

# Load the MNIST dataset
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Train the model
print("Training the model...")
train(X_train, y_train, X_test, y_test, epochs=10, learning_rate=0.01)