import numpy as np


class Node():
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        
        
class XGBoostRegressor():
    def __init__(self,n_estimators, learning_rate = 0.01, _lambda=1,
                 gamma = 0, max_depth=8, min_gain=0.05, min_samples_split=2):
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.min_samples_split = min_samples_split
        self.root = None
        self.compute_leaf_value = None
        self.information_gain = None
        self.alpha = _lambda
        self.gamma = gamma
        self.learning_rate = learning_rate
        
 
        
    def fit(self, X, y, depth=0) -> Node:
        # Initialize the root node
        n_features, n_samples = X.shape[1], X.shape[0]
        #  If the stopping criterion not met
        if n_samples > self.min_samples_split and depth < self.max_depth:
            # Find the best split
            best_split = self._best_split(X, y)
            if best_split['gain'] > self.min_gain:
                X_left, X_right, y_left, y_right = self._split(X, y, best_split['feature'], best_split['threshold'])
                left = self.fit(X_left, y_left, depth+1)
                right = self.fit(X_right, y_right, depth+1)
                return Node(feature=best_split['feature'], threshold=best_split['threshold'], left=left, right=right)
        
        # Return leaf node
        return Node(value=self._output_value(y))
    
    def _similarity_score(self, y):
        """
        Compute the similarity score for the leaf node.
        """
        denominator = (len(y) + self._lambda)
        score = np.sum(y, axis=0)**2 / denominator
        return score
        
    def _best_split(self, X, y):
        """
        Find the best split for the node.
        """
        best_split = {'gain': None, 'feature': None, 'threshold': None}
        n_features = X.shape[1]
        max_info_gain = -float('inf')
    
        # For each feature
        for i in range(n_features):
            # Possible thresholds are the quantiles of the feature values
            feature_values = X[:, i]
            thresholds = np.percentile(feature_values, np.arange(0, 100, 3))
            # Find the best threshold
            for threshold in thresholds:
                # Compute the score on each branch and the information gain
                y_left = y[X[:, i] < threshold]
                y_right = y[X[:, i] >= threshold]
                
                root_score = self.similarity_score(y)
                left_score = self.similarity_score(y_left)
                right_score = self.similarity_score(y_right)
                info_gain = left_score + right_score - root_score - self.gamma
                
                # If info gain is lower than 0, don't keep the split
                if info_gain < 0 :
                    continue
                # If info_gain is the best, update the best split
                if info_gain > max_info_gain
                    max_info_gain = info_gain
                    best_split['feature'] = i
                    best_split['threshold'] = threshold 

        # Return best split
        return best_split
    
    
    def _split(self, X, y, best_split: dict):
        """
        Split the data into left and right nodes.
        """
        left_mask = X[:, best_split.feature] < best_split.threshold
        right_mask = X[:, best_split.feature] >= best_split.threshold
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]
        
    def _output_value(self, y):
        """
        Compute the output value for the leaf node.
        output = - gradient / (hessian + lambda)
        """
        return np.sum(y) / (len(y) + self._lambda)
                
            
 
            
    