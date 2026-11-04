import numpy as np

# Training data
# Features: [Shape, Texture, Weight]
# Apple = 1, Orange = 0

X = np.array([
    [1, 1, 1],  # Apple
    [1, 1, 0],  # Apple
    [0, 0, 1],  # Orange
    [0, 1, 0],  # Orange
    [1, 0, 1],  # Applewhat
    [0, 0, 0]   # Orange
])

y = np.array([1, 1, 0, 0, 1, 0])

# Initialize weights and bias
weights = np.zeros(3)
bias = 0

learning_rate = 0.1

# Step activation function
def activation(x):
    return 1 if x >= 0 else 0

# Training the perceptron
for epoch in range(100):
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

# Display trained weights
print("Trained Weights:", weights)
print("Trained Bias:", bias)

# Testing with unseen fruit data
test_data = np.array([
    [1, 1, 1],  # Round, Smooth, Heavy
    [0, 0, 1],  # Elliptical, Rough, Heavy
    [1, 0, 0]   # Round, Rough, Light
])

print("\nTesting Results:")

for fruit in test_data:

    net = np.dot(fruit, weights) + bias
    result = activation(net)

    if result == 1:
        print(fruit, "-> Apple")
    else:
        print(fruit, "-> Orange")