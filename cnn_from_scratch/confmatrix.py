import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from cnn import SimpleCNN
from train import load_model

model = SimpleCNN()
load_model(model, "cnn_weights.npz")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Only use a subset of the data
X_train, y_train = X_train[:2000], y_train[:2000]
X_test, y_test = X_test[:200], y_test[:200]

# Normalize the images to [0, 1] range
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Add a channel dimension (28, 28) -> (28, 28, 1)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

y_pred = []
for i in range(len(X_test)):
    y_pred.append(np.argmax(model.forward(X_test[i])))
y_pred = np.array(y_pred)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
classes = unique_labels(y_test, y_pred)

# Plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

plt.show()