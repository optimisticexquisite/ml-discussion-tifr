import numpy as np

def softmax(x):
    shift_x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shift_x)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    
    epsilon = 1e-10
    return exp_x / (sum_exp_x + epsilon)


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)
