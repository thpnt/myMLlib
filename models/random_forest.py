import numpy as np
from myMLlib.models.decision_tree import TreeClassifier


class RandomForest:
    """
    A random forest classifier.

    Parameters:
    - n_trees (int): The number of decision trees in the random forest. Default is 100.
    - max_depth (int): The maximum depth of each decision tree. Default is None.
    - min_samples_split (int): The minimum number of samples required to split an internal node. Default is 2.
    - max_features (int): The number of features to consider when looking for the best split. Default is None.

    Methods:
    - fit(X, y): Fit the random forest to the training data.
    - predict(X): Make predictions on new data.

    """

    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        
    def fit(self, X, y):
        """
        Fit the random forest to the training data.

        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).
        - y (array-like): The target values of shape (n_samples,).

        Returns:
        - self: The fitted RandomForest object.

        """
        for _ in range(self.n_trees):
            # Bootstrap sample
            idxs = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_boot, y_boot = X[idxs], y[idxs]

            # Select random subset of features
            max_features = int(np.sqrt(X.shape[1])) if self.max_features is None else self.max_features
            feature_idxs = np.random.choice(X.shape[1], max_features, replace=False)
            X_boot = X_boot[:, feature_idxs]

            # Train a decision tree
            tree = TreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features)
            tree.fit(X_boot, y_boot)
            self.trees.append((tree, feature_idxs))
        return self
            
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).

        Returns:
        - predictions (array-like): The predicted target values of shape (n_samples,).

        """
        # Make predictions
        predictions = np.array([tree.predict(X[:, feature_idxs]) for tree, feature_idxs in self.trees])
        return np.round(np.mean(predictions, axis=0))
    
    
    