import numpy as np

# XOR input data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected output
y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
np.random.seed(42)

W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training
learning_rate = 0.5

for epoch in range(10000):

    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    predicted_output = sigmoid(final_input)

    # Calculate error
    error = y - predicted_output

    # Backpropagation
    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += learning_rate * hidden_output.T.dot(d_output)
    b2 += learning_rate * np.sum(d_output, axis=0, keepdims=True)

    W1 += learning_rate * X.T.dot(d_hidden)
    b1 += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

# Testing
print("XOR Results:")

for i in X:
    hidden = sigmoid(np.dot(i, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)

    result = 1 if output[0][0] >= 0.5 else 0
    print(i, "=", result)