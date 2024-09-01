import numpy as np
from cnn import SimpleCNN
def train(X_train, y_train, X_test, y_test, epochs, learning_rate):
    model = SimpleCNN()
    for epoch in range(epochs):
        loss = 0
        for i in range(len(X_train)):
            # Forward pass
            y_pred = model.forward(X_train[i])
            # Compute loss (cross-entropy)
            loss -= np.sum(y_train[i] * np.log(y_pred))
            # Backward pass
            model.backward(y_pred, y_train[i], learning_rate)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss/len(X_train)}')

        # Evaluate the model on the test set
        correct = 0
        for i in range(len(X_test)):
            y_pred = model.forward(X_test[i])
            if np.argmax(y_pred) == np.argmax(y_test[i]):
                correct += 1
        accuracy = correct / len(X_test)
        print(f'Accuracy: {accuracy * 100}%')
