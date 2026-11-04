import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize prototypes
n_features = X.shape[1]
n_prototypes = 3

prototypes = {}
proto_labels = {}

np.random.seed(42)

for c in np.unique(y_train):
    class_mean = np.mean(X_train[y_train == c], axis=0)

    for p in range(n_prototypes):
        prototypes[(c, p)] = (
            class_mean + 0.01 * np.random.randn(n_features)
        )
        proto_labels[(c, p)] = c

# Training parameters
learning_rate = 0.01
epochs = 50

# Train LVQ
for epoch in range(epochs):

    for xi, yi in zip(X_train, y_train):

        # Calculate distance from each prototype
        distances = {
            k: np.linalg.norm(xi - w)
            for k, w in prototypes.items()
        }

        # Find nearest prototype
        winner = min(distances, key=distances.get)

        winner_label = proto_labels[winner]

        # Update prototype
        if winner_label == yi:
            prototypes[winner] += learning_rate * (
                xi - prototypes[winner]
            )
        else:
            prototypes[winner] -= learning_rate * (
                xi - prototypes[winner]
            )

# Testing
y_pred = []

for xi in X_test:

    distances = {
        k: np.linalg.norm(xi - w)
        for k, w in prototypes.items()
    }

    winner = min(distances, key=distances.get)

    y_pred.append(proto_labels[winner])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", round(accuracy * 100, 2), "%")