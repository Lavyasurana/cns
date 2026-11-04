import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load dataset
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Separate target variable
death_event = data["DEATH_EVENT"]

# Remove target from training features
X = data.drop("DEATH_EVENT", axis=1)

# Handle missing values
X = X.fillna(X.mean())

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create SOM (10 x 10 grid)
som = MiniSom(
    x=10,
    y=10,
    input_len=X_scaled.shape[1],
    sigma=1.0,
    learning_rate=0.5,
    random_seed=42
)

# Initialize weights
som.random_weights_init(X_scaled)

# Train SOM
som.train_random(X_scaled, 1000)

print("SOM Training Completed")

# Find Best Matching Unit (BMU) for each patient
bmu_coordinates = np.array([
    som.winner(x) for x in X_scaled
])

# Get SOM neuron weights
weights = som.get_weights()

# Reshape weights for clustering
neuron_weights = weights.reshape(-1, X_scaled.shape[1])

# Apply KMeans to group SOM neurons into 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

neuron_clusters = kmeans.fit_predict(neuron_weights)

# Assign patients to clusters based on their BMU
patient_clusters = []

for x, y in bmu_coordinates:
    neuron_index = x * 10 + y
    patient_clusters.append(neuron_clusters[neuron_index])

patient_clusters = np.array(patient_clusters)

# Evaluate clustering
silhouette = silhouette_score(X_scaled, patient_clusters)

db_index = davies_bouldin_score(X_scaled, patient_clusters)

print("\nClustering Results")
print("Number of Clusters:", len(np.unique(patient_clusters)))
print("Silhouette Score:", round(silhouette, 4))
print("Davies-Bouldin Index:", round(db_index, 4))

# Compare clusters with DEATH_EVENT
results = pd.DataFrame({
    "Cluster": patient_clusters,
    "DEATH_EVENT": death_event.values
})

print("\nCluster vs Death Event:")
print(pd.crosstab(
    results["Cluster"],
    results["DEATH_EVENT"],
    margins=True
))

# Visualize SOM U-Matrix
plt.figure(figsize=(8, 6))

plt.imshow(som.distance_map(), cmap="bone_r")

plt.colorbar(label="Distance")
plt.title("SOM U-Matrix")
plt.xlabel("SOM X")
plt.ylabel("SOM Y")

plt.show()

# Visualize patient clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    bmu_coordinates[:, 0],
    bmu_coordinates[:, 1],
    c=patient_clusters,
    cmap="viridis"
)

plt.title("Heart Failure Patient Clusters")
plt.xlabel("SOM X")
plt.ylabel("SOM Y")
plt.colorbar(label="Cluster")

plt.show()

# Visualize average feature values by cluster
cluster_data = pd.DataFrame(X_scaled, columns=X.columns)
cluster_data["Cluster"] = patient_clusters

cluster_means = cluster_data.groupby("Cluster").mean()

plt.figure(figsize=(12, 6))

sns.heatmap(
    cluster_means,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Heatmap for Patient Clusters")
plt.show()