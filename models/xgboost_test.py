import numpy as np
from myMLlib.models.decision_tree import TreeRegressor, TreeClassifier, DecisionTree


class MeanSquaredError:
    """
    Mean squared error loss.
    """
    def gradient(self, y_true, y_pred):
        """
        Compute the gradient of the loss.
        """
        return -2 * (y_true - y_pred)

    def hessian(self, y_true, y_pred):
        """
        Compute the hessian of the loss.
        """
        return 2 * np.ones_like(y_true)

class XGBoostTree(DecisionTree):
        def __init__(self, n_estimators=100, learning_rate=0.1,_lambda=1, max_depth=8, loss=MeanSquaredError()):
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self._lambda = _lambda
            self.max_depth = max_depth
            self.trees = []
            self.init_pred = None
            self.loss = loss
        
        
        def similarity_score(self, gradient, hessian):
            """
            Compute the similarity score for the leaf node.
            """
            score = np.sum(gradient)**2 / (np.sum(hessian) + self._lambda)
            return score
        
        def gain_calculation(self, y, y1, y2):
            """
            Compute the gain of a particular split.
            """
            # Calculate the similarity score for the parent node
            sim_parent = self.similarity_score(y)
            
            # Calculate the similarity score for the child nodes
            sim_child1 = self.similarity_score(y1)
            sim_child2 = self.similarity_score(y2)
            
            # Calculate the information gain
            gain = sim_child1 + sim_child2 - sim_parent
            return gain
        
        def leaf_value(self, gradient, hessian):
            """
            Compute the output value for the leaf node.
            """
            return -np.sum(gradient) / (np.sum(hessian) + self._lambda)
        
        
    

class XGBoostRegressor():
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
        self.init_pred = np.full(shape=y.shape, fill_value=0.5)
        predictions[:, 0] = self.init_pred
        residuals[:, 0] = y - self.init_pred

        # Fit the trees
        # Loop to create the trees
        for m in range(self.n_estimators):
            # Initialize and fit the tree to residuals
            tree = XGBoostTree(max_depth=self.max_depth)
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
            
            
    