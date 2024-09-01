import numpy as np
from cnn import SimpleCNN
from tqdm import tqdm

def train(X_train, y_train, X_test, y_test, epochs, learning_rate, batch_size):
    model = SimpleCNN()
    num_samples = X_train.shape[0]

    for epoch in range(epochs):
        # Shuffle the training data
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        print(f'Epoch {epoch + 1}/{epochs}')

        # Mini-batch training
        for i in tqdm(range(0, num_samples, batch_size)):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            # Forward pass
            loss = 0
            for j in range(batch_size):
                y_pred = model.forward(X_batch[j])
                # Compute loss (cross-entropy)
                loss -= np.sum(y_batch[j] * np.log(y_pred))
                # Backward pass
                model.backward(y_pred, y_batch[j], learning_rate)
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss/num_samples}')

        # Evaluate the model on the test set
        correct = 0
        for i in range(len(X_test)):
            y_pred = model.forward(X_test[i])
            if np.argmax(y_pred) == np.argmax(y_test[i]):
                correct += 1
        accuracy = correct / len(X_test)
        print(f'Accuracy: {accuracy * 100}%')


def save_model(model, file_name="cnn_weights.npz"):
    # Create a dictionary to store weights and biases
    weights_dict = {}
    
    # For each layer, store its weights and biases
    if isinstance(model, SimpleCNN):
        weights_dict['conv1_kernels'] = model.conv1.kernels
        weights_dict['conv1_biases'] = model.conv1.biases
        weights_dict['fc1_weights'] = model.fc1.weights
        weights_dict['fc1_biases'] = model.fc1.biases
        weights_dict['fc2_weights'] = model.fc2.weights
        weights_dict['fc2_biases'] = model.fc2.biases
    
    # Save the dictionary to a file
    np.savez(file_name, **weights_dict)
    print(f'Model saved to {file_name}')

def load_model(model, file_name="cnn_weights.npz"):
    # Load the weights and biases from the file
    weights_dict = np.load(file_name)
    
    # For each layer, set its weights and biases
    if isinstance(model, SimpleCNN):
        model.conv1.kernels = weights_dict['conv1_kernels']
        model.conv1.biases = weights_dict['conv1_biases']
        model.fc1.weights = weights_dict['fc1_weights']
        model.fc1.biases = weights_dict['fc1_biases']
        model.fc2.weights = weights_dict['fc2_weights']
        model.fc2.biases = weights_dict['fc2_biases']

    print(f'Model loaded from {file_name}')

    


