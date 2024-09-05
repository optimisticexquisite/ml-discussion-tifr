import numpy as np
import tensorflow as tf
from utils import *
from models import Conv2D, MaxPool2D, Flatten, Dense
from cnn import SimpleCNN
from train import load_model
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the image data (pixel values from 0 to 1)
X_test = X_test.astype(np.float32) / 255.0

# Reshape the images to add the channel dimension for grayscale images
# Original shape is (batch_size, 28, 28), we reshape to (batch_size, 28, 28, 1)
X_test = np.expand_dims(X_test, axis=-1)

# Pick a random image from X_test
random_idx = np.random.randint(0, len(X_test))
random_image = X_test[random_idx]
random_label = y_test[random_idx]  # Corresponding true label for verification

# Load the model
model = SimpleCNN()
load_model(model, "cnn_weights.npz")

# Forward pass through the model
# random_image_batch = np.expand_dims(random_image, axis=0)  # Add batch dimension
y_pred = model.forward(random_image)

# Predicted class
predicted_class = np.argmax(y_pred)

# Show the randomly selected test image
plt.imshow(random_image.squeeze(), cmap='gray')
plt.title(f'Predicted Class: {predicted_class}, True Class: {random_label}')
plt.axis('off')
plt.show()

# Print the output probabilities and predicted class
print(f'Output probabilities: {y_pred}')
print(f'Predicted class: {predicted_class}')
