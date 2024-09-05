import numpy

weights = numpy.load("cnn_weights.npz")

conv1_kernels = weights['conv1_kernels']
print("Conv1 kernels shape:", conv1_kernels.shape)
conv1_biases = weights['conv1_biases']
print("Conv1 biases shape:", conv1_biases.shape)
fc1_weights = weights['fc1_weights']
print("FC1 weights shape:", fc1_weights.shape)
fc1_biases = weights['fc1_biases']
print("FC1 biases shape:", fc1_biases.shape)
fc2_weights = weights['fc2_weights']
print("FC2 weights shape:", fc2_weights.shape)
fc2_biases = weights['fc2_biases']

# Visualize the kernels of the first convolutional layer
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 4, figsize=(10, 5))
for i in range(8):
    ax = axs[i // 4, i % 4]
    ax.imshow(conv1_kernels[i, :, :, 0], cmap='gray')
    ax.axis('off')
    ax.set_title(f'Kernel {i+1}')

plt.tight_layout()
plt.show()
# Compare this snippet from cnn_from_scratch/visualize_weights.py:
