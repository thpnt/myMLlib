import numpy as np
from utils.utils import covariance, deflation_method


class PCA():
    """
    Parameters:
    - n_components (int): The number of principal components to keep.
    - n_iter (int, optional): The maximum number of iterations for the deflation method. Defaults to 1000.

    Attributes:
    - mean (ndarray): The mean of the input data.
    - std (ndarray): The standard deviation of the input data.
    - cov_matrix (ndarray): The covariance matrix of the input data.
    - n_iter (int): The maximum number of iterations for the deflation method.
    - eigmatrix (ndarray): The matrix of eigenvectors corresponding to the principal components.
    - explained_variance_ (float): The ratio of the total explained variance to the total variance.

    Methods:
    - fit(X): Fit the PCA model to the input data.
    - transform(X): Transform the input data using the fitted PCA model.

    Example usage:
    ```
    pca = PCA(n_components=2)
    pca.fit(X)
    X_transformed = pca.transform(X)
    ```
    """

    def __init__(self, n_components, n_iter=1000):
        self.n_components = n_components
        self.mean = None
        self.std = None
        self.cov_matrix = None
        self.n_iter = n_iter
        
    def fit(self, X) -> None:
        """
        Fit the PCA model to the input data.

        Parameters:
        - X (ndarray): The input data matrix of shape (n_samples, n_features).

        Returns:
        - None
        """
        # Stadardize the data
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = (X-self.mean)/self.std
        
        # Compute the covariance matrix
        self.cov_matrix = np.ndarray(shape=(X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                self.cov_matrix[i, j] = covariance(X[:, i], X[:, j])
        
        # Compute the eignenvectors and eigenvalues of the matrix
        eigenvectors, eigenvalues = deflation_method(self.cov_matrix, self.n_components, n_iter=self.n_iter)
        self.eigmatrix = eigenvectors[:, :self.n_components]
        eigenvalues = eigenvalues.flatten()
        
        # Compute the explained_variance
        self.explained_variance_ = np.sum(eigenvalues[:self.n_components]) / np.sum(eigenvalues)
        return 
    
    def transform(self, X) -> np.ndarray:
        """
        Transform the input data using the fitted PCA model.

        Parameters:
        - X (ndarray): The input data matrix of shape (n_samples, n_features).

        Returns:
        - ndarray: The transformed data matrix of shape (n_samples, n_components).
        """
        X = (X-self.mean)/self.std
        return np.dot(X, self.eigmatrix)