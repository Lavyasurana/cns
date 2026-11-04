import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Select one image of digit 0 and digit 1
patterns = []

for digit in [0, 1]:
    image = X_train[y_train == digit][0]

    # Convert image to binary (-1 and +1)
    binary = np.where(image > 127, 1, -1)

    # Convert 2D image into 1D vector
    patterns.append(binary.flatten())

patterns = np.array(patterns)

# Number of neurons
N = patterns.shape[1]

# Initialize weight matrix
W = np.zeros((N, N))

# Hebbian learning rule
for pattern in patterns:
    W += np.outer(pattern, pattern)

W = W / N

# Remove self-connections
np.fill_diagonal(W, 0)

print("Hopfield Network Trained")

# Add noise to a stored pattern
original = patterns[0].copy()

noisy = original.copy()

noise = np.random.rand(N) < 0.20

noisy[noise] *= -1

# Recall function
def recall(pattern, W, iterations=10):

    output = pattern.copy()

    for _ in range(iterations):

        # Update neurons in random order
        for i in np.random.permutation(len(output)):

            net = np.dot(W[i], output)

            output[i] = 1 if net >= 0 else -1

    return output

# Recall original pattern
reconstructed = recall(noisy, W)

# Calculate recall accuracy
accuracy = np.mean(reconstructed == original) * 100

print("Recall Accuracy:", round(accuracy, 2), "%")

# Display images
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(original.reshape(28, 28), cmap="gray")
plt.title("Original Digit")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy.reshape(28, 28), cmap="gray")
plt.title("Noisy Digit")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(reconstructed.reshape(28, 28), cmap="gray")
plt.title("Reconstructed Digit")
plt.axis("off")

plt.show()