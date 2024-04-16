import numpy as np

class Stump():
    """
    A decision stump classifier.

    Attributes:
        polarity (int): The polarity of the stump's predictions.
        feature_idx (int): The index of the feature used for splitting.
        threshold (float): The threshold value for splitting.
        total_error (float): The total error of the stump.
        alpha (float): The weight of the stump in the ensemble.

    Methods:
        stump_predict(X): Predicts the class labels for the given input samples.

    """

    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.total_error = None
        self.alpha = None
        
    def stump_predict(self, X):
        """
        Predicts the class labels for the given input samples.

        Args:
            X (numpy.ndarray): The input samples.

        Returns:
            numpy.ndarray: The predicted class labels.

        """
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.array([1 if X_column[i] > self.threshold else -1 for i in range(n_samples)])
        return predictions
        


class AdaBoost():
    """
    AdaBoost classifier implementation using decision stumps as weak learners.
    
    Parameters:
    -----------
    None
    
    Attributes:
    -----------
    stumps : list
        List of decision stumps used as weak learners.
    alphas : list
        List of alpha values corresponding to each decision stump.
    total_errors : list
        List of total errors at each iteration.
    """
    
    def __init__(self):
        self.stumps = []
        self.alphas = []
        self.total_errors = []
        
    def _init_weights(self, m):
        return np.ones(m) / m
    
    def _weighted_error(self, y, y_pred, weights):
        return sum(weights[y != y_pred])
    
    def _calculate_alpha(self, error):
        return 0.5 * np.log((1 - error) / error)
    
    def _update_weights(self, weights, alpha, y, y_pred):
        weights *= np.exp(-alpha * y * y_pred)
        return weights / sum(weights)
    
    def _compute_gini_index(self, y, weights):
        unique_classes, counts = np.unique(y, return_counts=True)
        class_weights = np.array([np.sum(weights[y == cls]) for cls in unique_classes])
        probabilities = class_weights / np.sum(weights)
        gini_index = 1 - np.sum(probabilities**2)
        return gini_index
    
    def fit(self, X, y, n_stumps):
        """
        Fit the AdaBoost classifier to the training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values.
        n_stumps : int
            The number of decision stumps to use as weak learners.
        
        Returns:
        --------
        None
        """
        n_samples, n_features = X.shape
        weights = self._init_weights(n_samples)

        for _ in range(n_stumps):
            # Create a new sample from X using the weights as a distribution
            indices = np.random.choice(np.arange(n_samples), size=n_samples, p=weights)
            X_sample = X[indices]
            y_sample = y[indices]
            stump = Stump()
            min_gini_index = -float('inf')

            for feature_idx in range(n_features):
                X_column = X_sample[:, feature_idx]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    left_indices = X_column <= threshold
                    right_indices = X_column > threshold
                    left_gini_index = self._compute_gini_index(y_sample[left_indices], weights[left_indices])
                    right_gini_index = self._compute_gini_index(y_sample[right_indices], weights[right_indices])
                    total_gini_index = (np.sum(left_indices) * left_gini_index + np.sum(right_indices) * right_gini_index) / n_samples

                    if total_gini_index < min_gini_index:
                        min_gini_index = total_gini_index
                        stump.feature_idx = feature_idx
                        stump.threshold = threshold

            y_pred = stump.stump_predict(X_sample)
            error = self._weighted_error(y_sample, y_pred, weights)
            alpha = self._calculate_alpha(error)
            weights = self._update_weights(weights, alpha, y_sample, y_pred)

            stump.alpha = alpha
            self.stumps.append(stump)
            
            
    def predict(self, X):
        """
        Predict the class labels for the input samples.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            The predicted class labels.
        """
        n_samples = X.shape[0]
        stump_predictions = np.array([stump.stump_predict(X) for stump in self.stumps])
        predictions = np.array([np.sum(stump_predictions[:, i] * np.array([stump.alpha for stump in self.stumps])) for i in range(n_samples)])
        return np.sign(predictions)
    