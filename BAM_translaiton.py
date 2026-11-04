import numpy as np

# English-Hindi dictionary
english = ["Dog", "Cat", "Sun"]
hindi = ["कुत्ता", "बिल्ली", "सूरज"]

# Convert words into bipolar binary encoding
def encode(index, size):

    vector = -np.ones(size)
    vector[index] = 1

    return vector

# Create input and output patterns
X = np.array([
    encode(i, len(english))
    for i in range(len(english))
])

Y = np.array([
    encode(i, len(hindi))
    for i in range(len(hindi))
])

# Initialize weight matrix
W = np.zeros((X.shape[1], Y.shape[1]))

# Hebbian learning rule
for i in range(len(X)):
    W += np.outer(X[i], Y[i])

print("BAM Network Trained")

# Activation function
def activation(x):
    return np.where(x >= 0, 1, -1)

# Forward recall: English -> Hindi
def forward_recall(x):

    y = activation(np.dot(x, W))

    return y

# Reverse recall: Hindi -> English
def reverse_recall(y):

    x = activation(np.dot(y, W.T))

    return x

# Find closest word pattern
def find_word(output, patterns, words):

    distances = [
        np.sum(output != pattern)
        for pattern in patterns
    ]

    index = np.argmin(distances)

    return words[index]

# Testing English to Hindi
print("\nEnglish to Hindi:")

for i in range(len(english)):

    output = forward_recall(X[i])

    result = find_word(output, Y, hindi)

    print(english[i], "->", result)

# Testing Hindi to English
print("\nHindi to English:")

for i in range(len(hindi)):

    output = reverse_recall(Y[i])

    result = find_word(output, X, english)

    print(hindi[i], "->", result)