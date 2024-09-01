import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from train import *
import numpy as np

# Load the MNIST dataset
import tensorflow as tf

if __name__ == '__main__':
    # Load the MNIST dataset from TensorFlow
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize the images to [0, 1] range
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Add a channel dimension (28, 28) -> (28, 28, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # One-hot encode the labels
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    # Train the model
    train(X_train, y_train, X_test, y_test, epochs=5, learning_rate=0.01, batch_size=32)

    # Save the model
    model = SimpleCNN()
    save_model(model, "cnn_weights.npz")
    print("Model saved to cnn_weights.npz")

    # Load the model
    # model = SimpleCNN()
    # load_model(model, "cnn_weights.npz")
    # print("Model loaded from cnn_weights.npz")



