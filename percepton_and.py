import numpy as np

# Step activation function
def activation(x):
    return 1 if x >= 0 else 0

# Perceptron training function
def train_perceptron(X, y):

    weights = np.zeros(X.shape[1])
    bias = 0
    learning_rate = 0.1

    # Training
    for epoch in range(20):
        for i in range(len(X)):

            # Calculate weighted sum
            net = np.dot(X[i], weights) + bias

            # Predict output
            prediction = activation(net)

            # Calculate error
            error = y[i] - prediction

            # Update weights and bias
            weights += learning_rate * error * X[i]
            bias += learning_rate * error

    return weights, bias


# AND Gate
X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_and = np.array([0, 0, 0, 1])


# OR Gate
X_or = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_or = np.array([0, 1, 1, 1])


# NOT Gate
X_not = np.array([
    [0],
    [1]
])

y_not = np.array([1, 0])


# Train all three gates
w_and, b_and = train_perceptron(X_and, y_and)
w_or, b_or = train_perceptron(X_or, y_or)
w_not, b_not = train_perceptron(X_not, y_not)


# Testing function
def test_gate(name, X, weights, bias):

    print("\n", name, "Gate Results:")

    for x in X:
        net = np.dot(x, weights) + bias
        output = activation(net)

        print(x, "->", output)


# Display results
test_gate("AND", X_and, w_and, b_and)
test_gate("OR", X_or, w_or, b_or)
test_gate("NOT", X_not, w_not, b_not)