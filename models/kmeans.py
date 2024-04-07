import numpy as np
from myMLlib.utils.utils import euclidean_distance

class KMeans():
    # Constructor
    def __init__(self, k_clusters=3, max_iter=1000, convergence_criterion=0.001):
        self.k_clusters = k_clusters
        self.max_iter = max_iter
    
    # Method to fit the model
    def fit(self, X) -> np.ndarray:
        # Initialize random centroids with shape (k_clusters, n_features)
        n_features = X.shape[1]
        centroids  = np.random.rand((self.k_clusters, n_features))
        
        # Iterate until convergence or max_iter
        for _ in range(self.max_iter): 
            # Assign clusters to data points based on the closest centroid
            # Create a list to store the data point index and the cluster index assigned to it
            clusters = np.zeros([X.shape[0], 2])
            for i,e in enumerate(X):
                # Compute the distance between the data point and each centroid
                distances = [euclidean_distance(e, centroid) for centroid in centroids]
                # Assign the data point to the cluster with the closest centroid
                cluster = np.argmin(distances)
                clusters[i, 0], clusters[i, 1] = i, cluster
            
            old_centroids = centroids.copy()
            # Update the centroids --> mean of the data points in each cluster
            for k in range(self.k_clusters):
                # Get the data points in cluster k
                data_points_index = clusters[clusters[:, 1] == k][:, 0]
                data_points = X[data_points_index]
                # Compute the mean of the data points in cluster k
                new_cluster = np.mean(data_points, axis=0)
                # Update the centroid of cluster k
                centroids[k] = new_cluster
                
            # Check for convergence
            if euclidean_distance(centroids, old_centroids) < self.convergence_criterion:
                self.centroids=centroids
                return self.centroids
        self.centroids = centroids
        return self.centroids
    
    # Method to predict the cluster of a new data point
    def predict(self, X) -> np.ndarray:
        # Assign cluster to each data point based on the closest centroid
        clusters = np.zeros([X.shape[0], 2])
        for i,e in enumerate(X):
            # Compute the distance between the data point and each centroid
            distances = [euclidean_distance(e, centroid) for centroid in self.centroids]
            # Assign the data point to the cluster with the closest centroid
            cluster = np.argmin(distances)
            clusters[i, 0], clusters[i, 1] = i, cluster
        return clusters[:, 1]
    