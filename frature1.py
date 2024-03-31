import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

for value in random_values:
    print("Sigmoid for", value, ":", sigmoid(value))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)


print("ReLU outputs:")
for value in random_values:
    print("ReLU for", value, ":", relu(value))

print("\nLeaky ReLU outputs:")
for value in random_values:
    print("Leaky ReLU for", value, ":", leaky_relu(value))

print("\nTanh outputs:")
for value in random_values:
    print("Tanh for", value, ":", tanh(value))
