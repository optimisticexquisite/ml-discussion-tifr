import numpy as np
from utils import *
import multiprocessing as mp

def convolve_single_depth(k, x, f, h_out, w_out, kernels, biases):
        output_k = np.zeros((h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                output_k[i, j] = np.sum(x[i:i+f, j:j+f, :] * kernels[k]) + biases[k]
        return k, output_k

def compute_gradients(k, d_output_k, x, f, h, w, kernels):
    d_kernels_k = np.zeros_like(kernels[k])
    d_x_k = np.zeros_like(x)
    
    for i in range(h - f + 1):
        for j in range(w - f + 1):
            d_kernels_k += d_output_k[i, j] * x[i:i+f, j:j+f, :]
            d_x_k[i:i+f, j:j+f, :] += d_output_k[i, j] * kernels[k]
    
    return k, d_kernels_k, d_x_k

def max_pool_single_slice(i, j, k, x, pool_h, pool_w):
    return i, j, k, np.max(x[i*pool_h:(i+1)*pool_h, j*pool_w:(j+1)*pool_w, k])

def max_pool_backward_single_slice(i, j, k, x, d_output, pool_h, pool_w):
    h_start, w_start = i * pool_h, j * pool_w
    slice_x = x[h_start:h_start + pool_h, w_start:w_start + pool_w, k]
    h_max, w_max = np.unravel_index(np.argmax(slice_x), slice_x.shape)
    
    d_x_local = np.zeros_like(x)
    d_x_local[h_start + h_max, w_start + w_max, k] = d_output[i, j, k]
    
    return d_x_local

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
        
        # Create a pool of workers
        pool = mp.Pool(mp.cpu_count())

        # Run convolve_single_depth in parallel using the pool
        results = [pool.apply_async(convolve_single_depth, args=(k, x, f, h_out, w_out, self.kernels, self.biases)) for k in range(d)]
        
        # Gather the results
        for result in results:
            k, output_k = result.get()
            output[:, :, k] = output_k

        pool.close()
        pool.join()

        return relu(output)

    def backward(self, d_output, learning_rate):
        h, w, c = self.input_shape
        f = self.kernel_size
        d = self.depth
        
        # Initialize gradients
        d_kernels = np.zeros_like(self.kernels)
        d_x = np.zeros_like(self.x)

        # Create a pool of workers for parallel processing
        pool = mp.Pool(mp.cpu_count())

        # Parallelize over the depth dimension
        results = [pool.apply_async(compute_gradients, args=(k, d_output[:, :, k], self.x, f, h, w, self.kernels)) for k in range(d)]

        # Collect results from each process
        for result in results:
            k, d_kernels_k, d_x_k = result.get()
            d_kernels[k] += d_kernels_k
            d_x += d_x_k

        pool.close()
        pool.join()

        # Update kernels
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
        
        # Create a pool of workers
        pool = mp.Pool(mp.cpu_count())

        # Apply parallelization over all indices of h_out, w_out, and d
        tasks = [(i, j, k, x, pool_h, pool_w) for i in range(h_out) for j in range(w_out) for k in range(d)]
        results = pool.starmap(max_pool_single_slice, tasks)

        # Collect the results
        for i, j, k, max_val in results:
            output[i, j, k] = max_val

        pool.close()
        pool.join()

        return output

    def backward(self, d_output):
        h, w, d = self.x.shape
        pool_h, pool_w = self.pool_size
        d_x = np.zeros_like(self.x)
        
        # Create a pool of workers
        pool = mp.Pool(mp.cpu_count())

        # Prepare tasks for parallelization
        tasks = [(i, j, k, self.x, d_output, pool_h, pool_w) for i in range(h // pool_h) for j in range(w // pool_w) for k in range(d)]

        # Use starmap to process each slice in parallel
        results = pool.starmap(max_pool_backward_single_slice, tasks)

        # Aggregate the results
        for result in results:
            d_x += result

        pool.close()
        pool.join()

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
