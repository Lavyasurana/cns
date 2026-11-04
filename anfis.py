import numpy as np
import pandas as pd

from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

# -----------------------------------
# STEP 1: Generate Dataset
# -----------------------------------

n_samples = 1200

temperature = np.random.uniform(10, 40, n_samples)
humidity = np.random.uniform(20, 95, n_samples)
occupancy = np.random.randint(0, 7, n_samples)
appliances = np.random.randint(0, 11, n_samples)
hour = np.random.randint(0, 24, n_samples)

morning_peak = np.exp(-((hour - 8) / 2.5) ** 2)
evening_peak = np.exp(-((hour - 20) / 3) ** 2)

energy = (
    0.025 * temperature
    + 0.006 * humidity
    + 0.11 * occupancy
    + 0.16 * appliances
    + 0.18 * morning_peak
    + 0.28 * evening_peak
    + 0.004 * np.maximum(temperature - 25, 0) * appliances
    + np.random.normal(0, 0.12, n_samples)
)

energy = np.clip(energy, 0, None)

df = pd.DataFrame({
    "Temperature": temperature,
    "Humidity": humidity,
    "Occupancy": occupancy,
    "Appliances": appliances,
    "HourOfDay": hour,
    "EnergyConsumption": energy
})

print("Dataset Preview:")
print(df.head())


# -----------------------------------
# STEP 2: Data Preprocessing
# -----------------------------------

X = df.drop("EnergyConsumption", axis=1).values
y = df["EnergyConsumption"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_inputs = X_train.shape[1]
n_mfs = 2


# -----------------------------------
# STEP 3: Initialize Membership Functions
# -----------------------------------

centers = np.tile([-0.7, 0.7], (n_inputs, 1))
widths = np.ones((n_inputs, n_mfs))


# Gaussian Membership Function
def gaussian(x, center, width):

    width = max(width, 0.001)

    return np.exp(
        -0.5 * ((x - center) / width) ** 2
    )


# -----------------------------------
# STEP 4: Generate Fuzzy Rules
# -----------------------------------

rules = np.array(
    list(product(range(n_mfs), repeat=n_inputs))
)

print("Total Fuzzy Rules:", len(rules))


# -----------------------------------
# STEP 5: Calculate Firing Strengths
# -----------------------------------

def calculate_firing(X, centers, widths):

    memberships = np.zeros(
        (len(X), n_inputs, n_mfs)
    )

    for j in range(n_inputs):
        for m in range(n_mfs):

            memberships[:, j, m] = gaussian(
                X[:, j],
                centers[j, m],
                widths[j, m]
            )

    firing = np.ones((len(X), len(rules)))

    for r, rule in enumerate(rules):

        for j, mf in enumerate(rule):

            firing[:, r] *= memberships[:, j, mf]

    normalized = firing / (
        firing.sum(axis=1, keepdims=True) + 1e-12
    )

    return normalized


# -----------------------------------
# STEP 6: Build Sugeno Design Matrix
# -----------------------------------

def build_design_matrix(X, normalized):

    X_bias = np.column_stack([
        np.ones(len(X)), X
    ])

    blocks = []

    for r in range(len(rules)):

        blocks.append(
            normalized[:, r:r+1] * X_bias
        )

    return np.concatenate(blocks, axis=1)


# -----------------------------------
# STEP 7: ANFIS Prediction
# -----------------------------------

def predict_anfis(X, centers, widths, theta=None):

    firing = calculate_firing(
        X, centers, widths
    )

    design = build_design_matrix(X, firing)

    if theta is None:
        return design

    return design @ theta


# -----------------------------------
# STEP 8: Train ANFIS Model
# -----------------------------------

learning_rate = 0.015
epochs = 12
epsilon = 0.001

# Least Squares Estimation
design = predict_anfis(
    X_train, centers, widths
)

theta = np.linalg.lstsq(
    design, y_train, rcond=1e-6
)[0]

for epoch in range(epochs):

    # Update consequent parameters
    design = predict_anfis(
        X_train, centers, widths
    )

    theta = np.linalg.lstsq(
        design, y_train, rcond=1e-6
    )[0]

    # Calculate base prediction error
    prediction = predict_anfis(
        X_train, centers, widths, theta
    )

    base_loss = np.mean(
        (prediction - y_train) ** 2
    )

    # Calculate gradients
    grad_centers = np.zeros_like(centers)
    grad_widths = np.zeros_like(widths)

    # Update centers
    for idx in np.ndindex(centers.shape):

        old = centers[idx]

        centers[idx] = old + epsilon

        p1 = predict_anfis(
            X_train, centers, widths, theta
        )

        loss1 = np.mean((p1 - y_train) ** 2)

        centers[idx] = old - epsilon

        p2 = predict_anfis(
            X_train, centers, widths, theta
        )

        loss2 = np.mean((p2 - y_train) ** 2)

        centers[idx] = old

        grad_centers[idx] = (
            loss1 - loss2
        ) / (2 * epsilon)

    # Update widths
    for idx in np.ndindex(widths.shape):

        old = widths[idx]

        widths[idx] = old + epsilon

        p1 = predict_anfis(
            X_train, centers, widths, theta
        )

        loss1 = np.mean((p1 - y_train) ** 2)

        widths[idx] = max(old - epsilon, 0.05)

        p2 = predict_anfis(
            X_train, centers, widths, theta
        )

        loss2 = np.mean((p2 - y_train) ** 2)

        widths[idx] = old

        grad_widths[idx] = (
            loss1 - loss2
        ) / (2 * epsilon)

    # Gradient descent
    centers -= learning_rate * np.clip(
        grad_centers, -1, 1
    )

    widths -= learning_rate * np.clip(
        grad_widths, -1, 1
    )

    widths = np.maximum(widths, 0.05)

    # Recalculate consequent parameters
    design = predict_anfis(
        X_train, centers, widths
    )

    theta = np.linalg.lstsq(
        design, y_train, rcond=1e-6
    )[0]

    prediction = predict_anfis(
        X_train, centers, widths, theta
    )

    loss = np.mean(
        (prediction - y_train) ** 2
    )

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"- Training MSE: {loss:.5f}"
    )

print("\nTraining Completed!")


# -----------------------------------
# STEP 9: Model Evaluation
# -----------------------------------

y_pred = predict_anfis(
    X_test, centers, widths, theta
)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))


# -----------------------------------
# STEP 10: Display Fuzzy Rules
# -----------------------------------

input_names = [
    "Temperature",
    "Humidity",
    "Occupancy",
    "Appliances",
    "HourOfDay"
]

mf_names = ["Low", "High"]

print("\nLearned Fuzzy Rules:")

for i, rule in enumerate(rules[:8]):

    conditions = [
        f"{input_names[j]} is {mf_names[rule[j]]}"
        for j in range(n_inputs)
    ]

    print(
        f"Rule {i+1}: IF "
        + " AND ".join(conditions)
        + " THEN Energy is a learned linear function"
    )


# -----------------------------------
# STEP 11: Test New Household
# -----------------------------------

new_household = np.array([[
    32, 65, 3, 6, 20
]])

new_scaled = scaler.transform(new_household)

predicted_energy = predict_anfis(
    new_scaled, centers, widths, theta
)

print(
    "\nPredicted Energy Consumption:",
    round(predicted_energy[0], 3),
    "kWh per time step"
)