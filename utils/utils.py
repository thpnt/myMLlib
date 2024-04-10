import numpy as np
import math

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def covariance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the covariance between two arrays.

    Args:
        x (np.ndarray): The first input array.
        y (np.ndarray): The second input array.

    Returns:
        float: The covariance between x and y.

    Raises:
        ValueError: If x and y do not have the same shape.

    """
    try:
        assert x.shape == y.shape
        x_mean = np.mean(x, axis=0)   
        y_mean = np.mean(y, axis=0)
        n = len(x)
        covariance = np.sum((x-x_mean)*(y-y_mean))/(n-1)
    
    except AssertionError:
        covariance = None
        raise ValueError("x and y must have the same shape")
        
    return covariance
    
    
def power_iteration_method(matrix: np.ndarray, n_iter: int, x_0:np.ndarray = None,convergence_criterion=1e-6):
    """
    Performs the power iteration method to find the dominant eigenvector and eigenvalue of a given matrix.

    Parameters:
    matrix (np.ndarray): The matrix for which to find the dominant eigenvector and eigenvalue.
    n_iter (int): The maximum number of iterations to perform.
    x_0 (np.ndarray, optional): The initial guess for the eigenvector. If not provided, a random vector will be used.
    convergence_criterion (float, optional): The convergence criterion for stopping the iteration. Defaults to 1e-6.

    Returns:
    eigenvector (np.ndarray): The dominant eigenvector of the matrix.
    eigenvalue (float): The dominant eigenvalue of the matrix.
    """
    
    if x_0 is None:
        x_0  = np.random.rand(matrix.shape[1])
    
    for i in range(n_iter):
        x = np.dot(matrix, x_0)
        x = x/np.linalg.norm(x)
        
        if np.linalg.norm(x - x_0) < convergence_criterion:
            break
        
        #Rayleigh quotient
        eigenvalue = np.dot(x.T, np.dot(matrix, x)) / np.dot(x.T, x)
        eigenvector = x_0.flatten()
        x_0 = x
            
    return eigenvector, eigenvalue
    
def deflation_method(matrix: np.ndarray, n_eigenpairs: int, n_iter: int = 1000):
    """
    Performs the deflation method for computing eigenpairs of a given matrix.

    Args:
        matrix (np.ndarray): The input matrix for which eigenpairs are to be computed.
        n_eigenpairs (int): The number of eigenpairs to compute.
        n_iter (int, optional): The maximum number of iterations for the power iteration method. Defaults to 1000.

    Raises:
        ValueError: If the number of eigenpairs is greater than the number of columns of the matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the computed eigenvectors and eigenvalues.
    """
    
    try:
        assert n_eigenpairs <= matrix.shape[1]
    except AssertionError:
        raise ValueError("The number of eigenpairs must be less than the number of columns of the matrix")
    
    eigenvectors = np.ndarray(shape=(matrix.shape[1], n_eigenpairs))
    eigenvalues = np.ndarray(shape=(n_eigenpairs, 1))
    x_0 = np.random.rand(matrix.shape[1], 1)
    
    for i in range(n_eigenpairs):
        eigenvectors[:, i], eigenvalues[i] = power_iteration_method(matrix, n_iter, x_0=x_0)
        matrix = matrix - eigenvalues[i]*np.outer(eigenvectors[:, i], eigenvectors[:, i])
    
    return eigenvectors, eigenvalues