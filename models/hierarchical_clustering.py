from utils import proximity_matrix
import numpy as np


class HierarchicalClustering():
    def __init__(self):
        self.linkage_matrix= None
        self.inertia = None
        
        
        
    def fit(self, X:np.ndarray):
        """
        Fit the HierarchicalClustering model to the given data.
        
        Parameters:
        - X: numpy.ndarray
            The input data to fit the model.
        
        Returns:
        - linkage_matrix: numpy.ndarray
            The linkage matrix of the clusters.
        """
        # Assign clusters to each data point
        n = X.shape[0]
        X[:, n+1] = np.arange(0, n, 1)
        self.linkage_matrix = []
        self.inertia = []
        
        # Compute proximity matrix
        proxmat = proximity_matrix(X[:,:-1])
        
        
        # Iterate until all data points are in the same cluster
        while len(np.unique(X[:, n+1])) > 1:
            # Find the two closest clusters
            exclude_indexes = []
            min_dist = np.argnanmin(proxmat)
            # If the two observation are in the same cluster already, find other value
            while X[min_dist[0, n+1]] == X[min_dist[1, n+1]]:
                exclude_indexes.append((min_dist[0], min_dist[1]))
                masked_proxmat = proxmat.copy().astype(float)

                for index in exclude_indexes:
                    masked_proxmat[index[0], index[1]] = np.nan

                min_dist = np.argnanmin(masked_proxmat)

            # Merge the two closest clusters
            # Find all indexes with same cluster
            cluster_2 = np.where(X[:, n+1] == X[min_dist[1], n+1])
            X[cluster_2, n+1] = X[min_dist[0], n+1]

            # Keep track of clusters and inertia
            self.linkage_matrix.append(X[:, n+1])
            # Compute inertia
            unique_clusters = np.unique(X[:, n+1])
            n_obs = [np.sum(X[:, n+1] == cluster for cluster in unique_clusters)]
            self.inertia.append(max(n_obs))
        
        return self.linkage_matrix, self.inertia
        
        