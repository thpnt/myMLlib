import math
import numpy as np

def euclidian_distance(u, v):
    """ Compute euclidian distance between 2 vectors"""
    d = 0
    for i in range(len(u)) :
        d += (u[i] - v[i])**2
    return math.sqrt(d)

def manhattan_distance(u, v) :
    """ Compute manhattan distance between 2 vectors"""
    d = 0
    for i in range(len(u)):
        d += abs(u[i] - v[i])
    return d


class KNN():
    """
    A class for K-Nearest Neighbors (KNN) algorithm, a non-parametric and instance-based machine learning technique.

    Parameters:
    - n_neighbors (int): The number of nearest neighbors to consider when making predictions.
    
    - euclidian_distance (booléan): A function or distance metric used to measure the similarity between data points. 
    Common choices include Euclidean distance or Manhattan distance.
    """
    
    def __init__(self, n_neighbors=5, euclidian_distance=True):
        self.n_neighbors = n_neighbors
        self.euclidian_distance = euclidian_distance
    
    def vote(self, neighbors_labels):
        """ Return the most common labels among the nearest neighbors"""
        count = np.bincount(neighbors_labels.as_type('int'))
        return count.argmax()
    
    def predict(self, X_train, X_test, y_train):
        #Instanciate prediction array
        y_pred = np.empty(X_test.shape[0])
        for i, x in enumerate(X_test):
            #compute distance of X_test points with every X_train points, and keep the k nearest
            if not self.euclidian_distance :
                idx = np.argsort([manhattan_distance(x, u) for u in X_train])[:self.k]
            else :
                idx = np.argsort([euclidian_distance(x, u) for u in X_train])[:self.k]
            #Get the labels of k nearest points
            k_nearest_neighbors = np.array([y_train[i] for i in idx])
            #Label our sample with a vote
            y_pred[i] = self.vote(k_nearest_neighbors)
        
        return y_pred