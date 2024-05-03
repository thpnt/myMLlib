import numpy as np

class MeanSquaredError():
    """
    MeanSquaredError is a class that represents the mean squared error loss function.
    It provides methods to calculate the loss, gradient, and hessian for the loss function.
    """

    def loss(self, y_true, y_pred):
        """
        Calculates the mean squared error loss.

        Parameters:
        - y_true: The true values.
        - y_pred: The predicted values.

        Returns:
        - The mean squared error loss.
        """
        return np.mean((y_true - y_pred)**2)/2
    
    def gradient(self, y_true, y_pred):
        """
        Calculates the gradient of the mean squared error loss.

        Parameters:
        - y_true: The true values.
        - y_pred: The predicted values.

        Returns:
        - The gradient of the mean squared error loss.
        """
        return - (y_true - y_pred)
    
    def hessian(self, y_true, y_pred):
        """
        Calculates the hessian of the mean squared error loss.

        Parameters:
        - y_true: The true values.
        - y_pred: The predicted values.

        Returns:
        - The hessian of the mean squared error loss.
        """
        return np.ones_like(y_true)
    

    
class Node():
    """
    Node is a class that represents a node in the XGBoost tree.
    Each node contains information about the feature index, threshold, value, and left and right child nodes.
    """

    def __init__(self, feature_index=None, threshold=None, value=None):
        """
        Initializes a Node object.

        Parameters:
        - feature_index: The index of the feature used for splitting at this node.
        - threshold: The threshold value used for splitting at this node.
        - value: The value associated with this node (used for leaf nodes).
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.left = None
        self.right = None

class XGBoostTree():
    """
    XGBoostTree is a class that represents a single decision tree in the XGBoost algorithm.
    """

    def __init__(self, _lambda=1, gamma=0, max_depth=32, min_samples_split=2):
        """
        Initialize the XGBoostTree object.

        Parameters:
        - _lambda (float): Regularization parameter lambda. Default is 1.
        - gamma (float): Minimum loss reduction required to make a further partition on a leaf node. Default is 0.
        - max_depth (int): Maximum depth of the tree. Default is 32.
        - min_samples_split (int): Minimum number of samples required to split an internal node. Default is 2.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._lambda = _lambda
        self.tree = None
        self.gamma = gamma

    def best_split(self, X, y):
        """
        Find the best split point for a given feature in the dataset.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): Target variable.

        Returns:
        - best_feature (int): Index of the best feature to split on.
        - best_threshold (float): Best threshold value for the split.
        - best_gain (float): Gain achieved by the best split.
        """
        best_feature, best_threshold, best_gain = None, None, 0
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.percentile(X[:, feature], np.arange(0, 100, 33))

            for threshold in thresholds:
                y_left = y[X[:, feature] < threshold]
                y_right = y[X[:, feature] >= threshold]

                # Compute the gain (with similarity scores)
                root_score = self.similarity_score(y)
                left_score = self.similarity_score(y_left)
                right_score = self.similarity_score(y_right)

                gain = left_score + right_score - root_score - self.gamma

                if gain > best_gain:
                    best_feature = feature
                    best_threshold = threshold
                    best_gain = gain

        return best_feature, best_threshold, best_gain

    def split(self, X, y):
        """
        Split the dataset based on the best split point.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): Target variable.

        Returns:
        - feature (int): Index of the feature used for the split.
        - threshold (float): Threshold value for the split.
        - X_left (numpy.ndarray): Subset of X on the left side of the split.
        - y_left (numpy.ndarray): Subset of y on the left side of the split.
        - X_right (numpy.ndarray): Subset of X on the right side of the split.
        - y_right (numpy.ndarray): Subset of y on the right side of the split.
        """
        feature, threshold, gain = self.best_split(X, y)

        if gain == 0:
            return None, None, None, None

        X_left = X[:, feature] < threshold
        y_left = y[X_left]
        X_right = X[:, feature] >= threshold
        y_right = y[X_right]

        return feature, threshold, X_left, y_left, X_right, y_right

    def build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): Target variable.
        - depth (int): Current depth of the tree.

        Returns:
        - node (Node): Root node of the built tree.
        """
        dataset = np.concatenate((X, y), axis=1)
        n_samples, n_features = X.shape

        # If stopping conditions not met
        if n_samples >= self.min_samples_split and depth <= self.max_depth:
            feature, threshold, X_left, y_left, X_right, y_right = self.split(X, y)

            if feature is not None:
                node = Node(feature, threshold)
                node.left = self.build_tree(X_left, y_left, depth+1)
                node.right = self.build_tree(X_right, y_right, depth+1)
                return node

        # Compute the leaf value
        return Node(value=self.compute_leaf_value(y))

    def fit(self, X, y):
        """
        Fit the XGBoostTree to the training data.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): Target variable.

        Returns:
        - None
        """
        self.tree = self.build_tree(X, y)
        return None

    def predict(self, X):
        """
        Predict the target variable for new input data.

        Parameters:
        - X (numpy.ndarray): Input features.

        Returns:
        - predictions (numpy.ndarray): Predicted target variable values.
        """
        def traverse(node, x):
            if node.value is not None:  # leaf node
                return node.value
            elif x[node.feature] < node.threshold:
                return traverse(node.left, x)
            else:  # x[node.feature] >= node.threshold
                return traverse(node.right, x)

        predictions = [traverse(self.tree, x) for x in X]
        return np.array(predictions)

    def similarity_score(self, y):
        """
        Compute the similarity score for a given target variable.

        Parameters:
        - y (numpy.ndarray): Target variable.

        Returns:
        - score (float): Similarity score.
        """
        return np.sum(y)**2 / (len(y) + self._lambda)

    def compute_leaf_value(self, y):
        """
        Compute the leaf value for the current node.

        Parameters:
        - y (numpy.ndarray): Target variable.

        Returns:
        - leaf_value (float): Leaf value.
        """
        return np.sum(y) / (len(y) + self._lambda)
    
    

    
class XGBoostRegressor():
    """ Constructor """
    def __init__(self, n_estimators=100, loss=MeanSquaredError(),
                 max_depth=32, min_samples_split=2,
                 learning_rate=0.1, _lambda=1, gamma=0):
        self.n_estimators = n_estimators
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self._lambda = _lambda
        self.gamma = gamma
        self.trees = []
        
    def fit(self, X, y):
        """
        Fit the gradient boosting regressor to the training data.

        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).
        - y (array-like): The target values of shape (n_samples,).

        Returns:
        None
        """
        # Initialize arrays of predicted residuals and terminal_regions (predictions)
        residuals = np.zeros(shape=(X.shape[0], self.n_estimators+1))
        predictions = np.zeros(shape=(X.shape[0], self.n_estimators+1))
        
        # Initial predictions = 0.5
        initial_prediction = np.ones_like(y) * 0.5
        predictions[:, 0] = initial_prediction
        
        # Compute initial residuals
        residuals[:, 0] = - self.loss.gradient(y, initial_prediction)
        
        # Fit the trees
        for m in range(self.n_estimators):
            # Fit a tree to the residuals
            tree = XGBoostTree(max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split)
            tree.fit(X, residuals[:,m])
            
            # make a new prediction and compute next residuals
            new_prediction = predictions[:, m] + self.learning_rate * tree.predict(X)
            predictions[:, m+1] = new_prediction
            residuals[:, m+1] = - self.loss.gradient(y, new_prediction)
            
            # Save the tree
            self.trees.append(tree)
            
        return None
    
    
    def predict(self, X):
        """
        Predict the target variable for new input data.

        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).

        Returns:
        - prediction (array-like): The predicted target values of shape (n_samples,).
        """
        # Initialize the prediction with the mean of the target variable
        prediction = np.ones(X.shape[0]) * 0.5
        
        # Add the prediction of each tree
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)
            
        return prediction
            
        
    
    
    