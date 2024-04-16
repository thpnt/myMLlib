import numpy as np
from myMLlib.models.decision_tree import TreeRegressor, TreeClassifier

# Gradientboost regressor

class GradientBoostRegressor():
    """
    A gradient boosting regressor implementation.

    Parameters:
    - n_estimators (int): The number of boosting stages to perform. Default is 100.
    - learning_rate (float): The learning rate of the boosting process. Default is 0.1.
    - max_depth (int): The maximum depth of the individual regression trees. Default is 8.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = None

    def fit(self, X, y):
        """
        Fit the gradient boosting regressor to the training data.

        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).
        - y (array-like): The target values of shape (n_samples,).

        Returns:
        None
        """
        # Initialize arrays of predicted residuals and terminal_regions
        residuals = np.zeros(shape=(X.shape[0], self.n_estimators+1))
        predictions = np.zeros(shape=(X.shape[0], self.n_estimators+1))

        # Initialize the model with the mean of the target variable
        self.init_pred = np.mean(y)
        predictions[:, 0] = self.init_pred
        residuals[:, 0] = y - self.init_pred

        # Fit the trees
        # Loop to create the trees
        for m in range(self.n_estimators):
            # Initialize and fit the tree to residuals
            tree = TreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals[:, m])

            # Predict the residuals for the current tree
            predicted_residuals = tree.predict(X)

            # Predict the target variable for the current tree
            new_pred = predictions[:, m] + self.learning_rate * predicted_residuals
            predictions[:, m+1] = new_pred

            # Compute new residuals
            new_residuals = y - new_pred
            residuals[:, m+1] = new_residuals

            # Save the tree
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the target variable for new input data.

        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).

        Returns:
        - prediction (array-like): The predicted target values of shape (n_samples,).
        """
        # Initialize the prediction with the mean of the target variable
        prediction = np.mean(self.init_pred)

        # Add the predictions of each tree
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)

        return prediction

    def mse_score(self, X, y):
        """
        Compute the mean squared error (MSE) score for the regressor on the given data.

        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).
        - y (array-like): The target values of shape (n_samples,).

        Returns:
        - mse_score (float): The mean squared error score.
        """
        y_pred = self.predict(X)
        return np.mean((y - y_pred)**2)
            
            
    
    
# Gradientboost classifier
class GradientBoostClassifier():
    """
    A gradient boosting classifier implementation.

    Parameters:
    - n_estimators (int): The number of boosting stages to perform. Default is 100.
    - learning_rate (float): The learning rate of the boosting process. Default is 0.1.
    - max_depth (int): The maximum depth of the individual classification trees. Default is 8.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = None

    def fit(self, X, y):
        """
        Fit the gradient boosting classifier to the training data.

        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).
        - y (array-like): The target values of shape (n_samples,).

        Returns:
        None
        """
        # Initialize arrays of predicted residuals and terminal_regions
        residuals = np.zeros(shape=(X.shape[0], self.n_estimators+1))
        predictions = np.zeros(shape=(X.shape[0], self.n_estimators+1))

        # Initialize the model with the mean of the target variable
        self.init_pred = np.mean(y)
        predictions[:, 0] = self.init_pred
        residuals[:, 0] = y - self.init_pred

        # Fit the trees
        # Loop to create the trees
        for m in range(self.n_estimators):
            # Initialize and fit the tree to residuals
            tree = TreeClassifier(max_depth=self.max_depth)
            tree.fit(X, residuals[:, m])

            # Predict the residuals for the current tree
            predicted_residuals = tree.predict(X)

            # Predict the target variable for the current tree
            new_pred = predictions[:, m] + self.learning_rate * predicted_residuals
            predictions[:, m+1] = new_pred

            # Compute new residuals
            new_residuals = y - new_pred
            residuals[:, m+1] = new_residuals

            # Save the tree
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the target variable for new input data.

        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).

        Returns:
        - prediction
        """
        # Initialize the prediction with the mean of the target variable
        prediction = np.mean(self.init_pred)

        # Add the predictions of each tree
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)

        return prediction
    
    def accuracy(self, X, y):
        """
        Compute the accuracy score for the classifier on the given data.
    
        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).
        - y (array-like): The target values of shape (n_samples,).
    
        Returns:
        - accuracy_score (float): The accuracy score.
        """
        y_pred = self.predict(X)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        return np.mean(y == y_pred)