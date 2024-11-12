import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class kMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        # Random initialization of centroids
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]

        for _ in range(self.max_iter):
            # Assign clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids.copy()
            # Move centroids
            self.centroids = self.move_centroids(X, cluster_group)
            # Check for convergence
            if np.array_equal(old_centroids, self.centroids):
                break

        return cluster_group

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def move_centroids(self, X, cluster_group):
        new_centroids = []
        for i in range(self.n_clusters):
            if np.any(cluster_group == i):  # Check if any points are assigned to this cluster
                new_centroids.append(X[cluster_group == i].mean(axis=0))
            else:
                # Retain old centroid if no points are assigned
                new_centroids.append(self.centroids[i])  
        return np.array(new_centroids)

    def inertia(self, X, cluster_group):
        # Calculate inertia (sum of squared distances to the nearest centroid)
        return sum(np.linalg.norm(X[cluster_group == i] - self.centroids[i]) ** 2 for i in range(self.n_clusters))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Parameters
n_samples = 6000
n_features = 2
initial_centers = 4
std_dev = 0.9

# Generate random data
X, _ = make_blobs(n_samples=n_samples, centers=initial_centers, cluster_std=std_dev, random_state=42)

# a) Scatter plot of the randomly generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=5, alpha=0.5)
plt.title("Scatter Plot of Randomly Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


def find_optimal_clusters(X, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = kMeans(n_clusters=k)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia(X, labels))
    
    return inertias


max_k = 10
inertias = find_optimal_clusters(X, max_k)

    # Plot the inertia values to visualize the elbow
plt.figure(figsize=(8, 6))
plt.plot(range(1, max_k + 1), inertias, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters (K-Means)')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range(1, max_k + 1))
plt.grid()
plt.show()

km = kMeans(n_clusters=4)
y_pred = km.fit_predict(X)

import pandas as pd

df = pd.DataFrame()

df['col1'] = X[:,0]
df['col2'] = X[:,1]

df['label'] = y_pred

import plotly.express as px
# Create a 2D scatter plot with hue based on the label
fig = px.scatter(df, x='col1', y='col2', color='label', title='2D Scatter Plot with Hue')

# Show the plot
fig.show()

print("From the WCSS graph , it is evident that the optimal number of clusters is 4")


km = kMeans(n_clusters=3)
y_pred = km.fit_predict(X)

df = pd.DataFrame()

df['col1'] = X[:,0]
df['col2'] = X[:,1]

df['label'] = y_pred

import plotly.express as px
# Create a 2D scatter plot with hue based on the label
fig = px.scatter(df, x='col1', y='col2', color='label', title='scatter plot with 3 clusters')

# Show the plot
fig.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


# a) Read the dataset from the CSV file
data = pd.read_csv("countries_continents.csv")

# Display the first few rows of the dataset for reference
print(data.head())

# b) Get the unique continents from the dataset
unique_continents = data['Continent'].unique()
num_continents = len(unique_continents)
print(f"Unique continents: {unique_continents}")
print(f"Number of unique continents: {num_continents}")

# c) Map text data to numbers (Label encoding)
label_encoder = LabelEncoder()
data['Continent'] = label_encoder.fit_transform(data['Continent'])
data['Country'] = label_encoder.fit_transform(data['Country'])




# Assuming the dataset has numeric features for clustering
# Extracting relevant features (you may need to adjust this based on your dataset)
features = data[['Country', 'Latitude', 'Longitude', 'Continent']].values  # Adjust column names as necessary


wcss = []
for i in range(1,20):
    km = KMeans(n_clusters=i)
    km.fit_predict(features)
    wcss.append(km.inertia_)

plt.plot(range(1,20),wcss)

# Running K-Means with the number of unique continents
kmeans = KMeans(n_clusters=8, max_iter=100)
data['Cluster'] = kmeans.fit_predict(features)

# Plotting the clustering results
plt.figure(figsize=(10, 6))
plt.scatter(features[:, 0], features[:, 1], c=data['Cluster'], cmap='viridis', s=50)
plt.title(f"K-Means Clustering with 6 Clusters")
plt.xlabel("Feature 1 (Population)")  # Replace with actual feature name
plt.ylabel("Feature 2 (GDP)")  # Replace with actual feature name
plt.colorbar(label="Cluster Label")
plt.show()



# Observations
print("Observations:")
print("As the number of clusters changes, the cluster formation and separation may vary.")
print("From Wcss we can observe that the elbow point occurs at around 6 clusters")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class kMeans_P:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        # K-Means++ Initialization
        self.centroids = self.kmeans_plus_plus_init(X)

        for i in range(self.max_iter):
            # Assign clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids.copy()
            # Move centroids
            self.centroids = self.move_centroids(X, cluster_group)
            # Check for convergence
            if np.array_equal(old_centroids, self.centroids):
                break

        return cluster_group

    def kmeans_plus_plus_init(self, X):
        # Choose the first centroid randomly
        random_index = random.randint(0, X.shape[0] - 1)
        centroids = [X[random_index]]

        for _ in range(1, self.n_clusters):
            # Calculate distances from the existing centroids
            distances = np.array([min([np.linalg.norm(x - centroid) ** 2 for centroid in centroids]) for x in X])
            # Select the next centroid based on a probability distribution
            probabilities = distances / distances.sum()
            next_index = np.random.choice(len(X), p=probabilities)
            centroids.append(X[next_index])

        return np.array(centroids)

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def move_centroids(self, X, cluster_group):
        new_centroids = []
        for i in range(self.n_clusters):
            if np.any(cluster_group == i):  # Check if any points are assigned to this cluster
                new_centroids.append(X[cluster_group == i].mean(axis=0))
            else:
                # Retain old centroid if no points are assigned
                new_centroids.append(self.centroids[i])  
        return np.array(new_centroids)

    def inertia(self, X, cluster_group):
        # Calculate inertia (sum of squared distances to the nearest centroid)
        return sum(np.linalg.norm(X[cluster_group == i] - self.centroids[i]) ** 2 for i in range(self.n_clusters))

# Function to find the optimal number of clusters using the Elbow Method
def find_optimal_clusters(X, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = kMeans_P(n_clusters=k)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia(X, labels))
    
    return inertias

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    n_samples = 6000
    n_features = 2
    X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.9, random_state=42)

    # Find optimal number of clusters
    max_k = 10
    inertias = find_optimal_clusters(X, max_k)

    # Plot the inertia values to visualize the elbow
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.xticks(range(1, max_k + 1))
    plt.grid()
    plt.show()


# Fit K-Means with 4 clusters
kmeans = kMeans_P(n_clusters=4)
labels = kmeans.fit_predict(X)

    # Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=5, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering with 4 Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

    # Optional: Find and plot inertia values
max_k = 10
inertias = find_optimal_clusters(X, max_k)

    # Plot the inertia values to visualize the elbow
plt.figure(figsize=(8, 6))
plt.plot(range(1, max_k + 1), inertias, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range(1, max_k + 1))
plt.grid()
plt.show()





