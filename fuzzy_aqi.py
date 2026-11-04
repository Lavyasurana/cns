import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Define AQI fuzzy sets
# -----------------------------------

aqi = np.linspace(0, 500, 200)

# Membership functions
def gaussian(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

G = gaussian(aqi, 30, 30)       # Good
M = gaussian(aqi, 100, 40)      # Moderate
P = gaussian(aqi, 200, 50)      # Poor
S = gaussian(aqi, 350, 70)      # Severe

# Plot AQI fuzzy sets
plt.figure(figsize=(9, 5))

plt.plot(aqi, G, label="Good")
plt.plot(aqi, M, label="Moderate")
plt.plot(aqi, P, label="Poor")
plt.plot(aqi, S, label="Severe")

plt.xlabel("Air Quality Index (AQI)")
plt.ylabel("Membership Value")
plt.title("Fuzzy Sets for AQI Levels")
plt.legend()
plt.grid()
plt.show()


# -----------------------------------
# 2. Fuzzy Operations
# -----------------------------------

# Union using Maximum
union = np.maximum(G, S)

# Intersection using Minimum
intersection = np.minimum(G, S)

# Bounded Sum
bounded_sum = np.minimum(G + S, 1)

# Bounded Difference
bounded_difference = np.maximum(G - S, 0)

# Display fuzzy operations
plt.figure(figsize=(10, 5))

plt.plot(aqi, G, label="Good")
plt.plot(aqi, S, label="Severe")
plt.plot(aqi, union, label="Union")
plt.plot(aqi, intersection, label="Intersection")
plt.plot(aqi, bounded_sum, label="Bounded Sum")
plt.plot(aqi, bounded_difference, label="Bounded Difference")

plt.xlabel("AQI")
plt.ylabel("Membership Value")
plt.title("Fuzzy Operations")
plt.legend()
plt.grid()
plt.show()


# -----------------------------------
# 3. Define Fuzzy Sets A, B and C
# -----------------------------------

# Set A: AQI Levels
# Good, Moderate, Poor, Severe

A = np.array([1.0, 0.7, 0.3, 0.1])

# Set B: Health Effects
# Healthy, Mild Symptoms, Serious Symptoms

B = np.array([1.0, 0.6, 0.2])

# Set C: Desired Health Conditions
# Healthy, Mild Symptoms, Serious Symptoms

C = np.array([1.0, 0.5, 0.1])

print("\nSet A (AQI Levels):", A)
print("Set B (Health Effects):", B)
print("Set C (Desired Health):", C)


# -----------------------------------
# 4. Cartesian Product
# -----------------------------------

# Relation R between A and B
R = np.minimum(A.reshape(-1, 1), B)

print("\nCartesian Product R (A x B):")
print(R)

# Relation S between B and C
S_relation = np.minimum(B.reshape(-1, 1), C)

print("\nCartesian Product S (B x C):")
print(S_relation)


# -----------------------------------
# 5. Max-Min Composition
# -----------------------------------

def max_min_composition(R, S):
    result = np.zeros((R.shape[0], S.shape[1]))

    for i in range(R.shape[0]):
        for j in range(S.shape[1]):
            values = []

            for k in range(R.shape[1]):
                values.append(min(R[i][k], S[k][j]))

            result[i][j] = max(values)

    return result


max_min = max_min_composition(R, S_relation)

print("\nMax-Min Composition:")
print(max_min)


# -----------------------------------
# 6. Max-Product Composition
# -----------------------------------

def max_product_composition(R, S):
    result = np.zeros((R.shape[0], S.shape[1]))

    for i in range(R.shape[0]):
        for j in range(S.shape[1]):

            values = []

            for k in range(R.shape[1]):
                values.append(R[i][k] * S[k][j])

            result[i][j] = max(values)

    return result


max_product = max_product_composition(R, S_relation)

print("\nMax-Product Composition:")
print(max_product)


# -----------------------------------
# 7. Display Results
# -----------------------------------

print("\nFinal Results:")

print("\nMax-Min Result:")
print(np.round(max_min, 2))

print("\nMax-Product Result:")
print(np.round(max_product, 2))


# -----------------------------------
# 8. Visualize Relations
# -----------------------------------

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(R, cmap="viridis", vmin=0, vmax=1)
plt.title("Relation R (A x B)")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(max_min, cmap="viridis", vmin=0, vmax=1)
plt.title("Max-Min Composition")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(max_product, cmap="viridis", vmin=0, vmax=1)
plt.title("Max-Product Composition")
plt.colorbar()

plt.tight_layout()
plt.show()