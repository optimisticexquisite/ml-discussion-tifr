import numpy as np
from utils import *
class Conv2D:
    def __init__(self, input_shape, kernel_size, depth):
        self.input_shape = input_shape  # (height, width, channels)
        self.kernel_size = kernel_size  # filter size (f, f)
        self.depth = depth  # number of filters
        self.kernels = np.random.randn(depth, kernel_size, kernel_size, input_shape[2]) * 0.01
        self.biases = np.zeros((depth, 1))

    def forward(self, x):
        self.x = x
        h, w, c = self.input_shape
        f = self.kernel_size
        d = self.depth
        h_out = h - f + 1
        w_out = w - f + 1
        output = np.zeros((h_out, w_out, d))
        for k in range(d):
            for i in range(h_out):
                for j in range(w_out):
                    output[i, j, k] = np.sum(x[i:i+f, j:j+f, :] * self.kernels[k]) + self.biases[k]
        return relu(output)

    def backward(self, d_output, learning_rate):
        h, w, c = self.input_shape
        f = self.kernel_size
        d = self.depth
        d_kernels = np.zeros_like(self.kernels)
        d_x = np.zeros_like(self.x)
        for k in range(d):
            for i in range(h - f + 1):
                for j in range(w - f + 1):
                    d_kernels[k] += d_output[i, j, k] * self.x[i:i+f, j:j+f, :]
                    d_x[i:i+f, j:j+f, :] += d_output[i, j, k] * self.kernels[k]
        self.kernels -= learning_rate * d_kernels
        return d_x

class MaxPool2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, x):
        self.x = x
        h, w, d = x.shape
        pool_h, pool_w = self.pool_size
        h_out = h // pool_h
        w_out = w // pool_w
        output = np.zeros((h_out, w_out, d))
        for i in range(h_out):
            for j in range(w_out):
                for k in range(d):
                    output[i, j, k] = np.max(x[i*pool_h:(i+1)*pool_h, j*pool_w:(j+1)*pool_w, k])
        return output

    def backward(self, d_output):
        h, w, d = self.x.shape
        pool_h, pool_w = self.pool_size
        d_x = np.zeros_like(self.x)
        for i in range(h // pool_h):
            for j in range(w // pool_w):
                for k in range(d):
                    (h_start, w_start) = (i * pool_h, j * pool_w)
                    slice_x = self.x[h_start:h_start+pool_h, w_start:w_start+pool_w, k]
                    h_max, w_max = np.unravel_index(np.argmax(slice_x), slice_x.shape)
                    d_x[h_start + h_max, w_start + w_max, k] = d_output[i, j, k]
        return d_x

class Flatten:
    def forward(self, x):
        self.x_shape = x.shape
        return x.flatten().reshape(1, -1)
    
    def backward(self, d_output):
        return d_output.reshape(self.x_shape)

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, d_output, learning_rate):
        d_weights = np.dot(self.x.T, d_output)
        d_x = np.dot(d_output, self.weights.T)
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
        return d_x
