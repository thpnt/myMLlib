import numpy as np

class Node():
    """
    Represents a node in a decision tree.
    
    Attributes:
        feature (str): The feature used for splitting at this node.
        threshold (float): The threshold value used for splitting at this node.
        left (Node): The left child node.
        right (Node): The right child node.
        value (float or int): The predicted value at this leaf node.
    """
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # Decision Node attributes
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        
        # Leaf Node attributes
        self.value = value
    
    
    
class DecisionTree():
    """
    A decision tree classifier implementation.

    Parameters:
    -----------
    max_depth : int, optional
        The maximum depth of the decision tree. If None, the tree will expand until all leaves are pure or until all leaves contain less than `min_samples_split` samples.
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node. Default is 2.
    min_gain : float, optional
        The minimum impurity calculation required to split an internal node. Default is 0.

    Attributes:
    -----------
    root : Node
        The root node of the decision tree.

    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=0, max_features=None):
        ''' Constructor '''
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.root = None
        self.impurity_calculation = None
        self.leaf_value_calculation = None
        self.score_calculation = None
        self.max_features = max_features
        
        
    def build_tree(self, dataset, depth=0) -> Node:
        """
        Recursively builds the decision tree.
        Parameters:
        -----------
        dataset : numpy.ndarray
            The training dataset. The last column should represent the labels.
        depth : int
            The current depth of the tree.
        
        Returns:
        --------
        Node
            The root node of the built tree.
        """
        # Initialize the root node
        n_samples, n_features = dataset.shape
        # Check for stopping conditions
        if n_samples >= self.min_samples_split or depth <= self.max_depth:
            # Find the best split
            best_split = self.best_split(dataset)
            # If impurity calculation is superior to the threshold, split the dataset
            if best_split[2] > self.min_gain:
                # Recursively build the left and right subtrees
                left_subset, right_subset, dataset = self.split(dataset, best_split[0], best_split[1])
                left_subtree = self.build_tree(left_subset, depth+1)
                right_subtree = self.build_tree(right_subset, depth+1)
                return Node(best_split[0], best_split[1], left_subtree, right_subtree)
        
        # Compute leaf node
        return Node(value=self.leaf_value_calculation(dataset))
    
    def fit(self, X, y) -> None:
        """
        Fit the decision tree classifier to the training data.
        Parameters:
        -----------
        X : numpy.ndarray
            The training input samples. Shape (n_samples, n_features).
        y : numpy.ndarray
            The target values. Shape (n_samples,).
        """
        dataset = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        self.root = self.build_tree(dataset)
        return
    
    def predict(self, X) -> np.ndarray:
        """
        Predict the class labels for the input samples.
        Parameters:
        -----------
        X : numpy.ndarray
            The input samples. Shape (n_samples, n_features).
        Returns:
        --------
        numpy.ndarray
            The predicted class labels. Shape (n_samples,).
        
        """
        predictions = []
        for x in X:
            node = self.root
            while node.value is None:
                if x[node.feature] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value)
        return np.array(predictions)

    
    
    def best_split(self, dataset) -> tuple:
        """
        Find the best split for a dataset.
        Parameters:
        -----------
        dataset : numpy.ndarray
            The dataset for which to find the best split.
        Returns:
        --------
        tuple
            A tuple containing the index of the best feature to split on, the threshold value to split on, 
            and the impurity gain of the split.
        """
        best_feature, best_threshold = None, None
        n_features = len(dataset[0]) -1
        max_info_gain = -float("inf")
        
        
        if self.max_features is None:
            for i in range(n_features):
                for threshold in np.unique(dataset[:, i]):
                    info_gain = self.impurity_calculation(dataset, i, threshold)
                    if info_gain > max_info_gain:
                        best_feature = i
                        best_threshold = threshold
                        max_info_gain = info_gain
        elif self.max_features is not None:
            subset = np.random.choice(n_features, self.max_features, replace=False)
            for i in subset:
                for threshold in np.unique(dataset[:, i]):
                    info_gain = self.impurity_calculation(dataset, i, threshold)
                    if info_gain > max_info_gain:
                        best_feature = i
                        best_threshold = threshold
                        max_info_gain = info_gain
                    
        return best_feature, best_threshold, max_info_gain
    
    def split(self, dataset, feature_index, threshold) -> tuple:
        """
        Split a dataset into two subsets.
        Parameters:
        -----------
        dataset : numpy.ndarray
            The dataset to split.
        feature_index : int
            The index of the feature to split on.
        threshold : float
            The threshold value to split the feature on.
        Returns:
        --------
        tuple
            A tuple containing the left subset, the right subset, and the original dataset.
        
        """
        left_subset = dataset[dataset[:, feature_index] < threshold]
        right_subset = dataset[dataset[:, feature_index] >= threshold]
        return left_subset, right_subset, dataset
    

        
        
class TreeClassifier(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=0):
        super().__init__(max_depth, min_samples_split, min_gain, max_features=None)
        self.impurity_calculation = self.information_gain
        self.leaf_value_calculation = self.most_common_label
        self.score_calculation = self.accuracy_score
        
    def compute_entropy(self,dataset) -> float:
        """
        Compute the entropy of a dataset.
        Parameters:
        -----------
        dataset : numpy.ndarray
            The dataset for which to compute the entropy.
        Returns:
        --------
        float
            The entropy of the dataset.
        """
        y = dataset[:, -1]
        count = len(y)
        classes = np.unique(y)
        entropy = 0
        for cls in classes :
            p = np.sum(y==cls) / count
            entropy -= p * np.log2(p)
        return entropy
    
    def information_gain(self, dataset, feature_index, threshold) -> float:
        """
        Compute the information gain of a dataset given a feature and threshold.
        Parameters:
        -----------
        dataset : numpy.ndarray
            The dataset for which to compute the information gain.
        feature_index : int
            The index of the feature to split on.
        threshold : float
            The threshold value to split the feature on.
        Returns:
        --------
        float
            The information gain of the dataset given the feature and threshold. 
            
        """
        e_0, n = self.compute_entropy(dataset), len(dataset)
        e_1_left, n_left = self.compute_entropy(dataset[dataset[:, feature_index] < threshold]), len(dataset[dataset[:, feature_index] < threshold])
        e_1_right, n_right = self.compute_entropy(dataset[dataset[:, feature_index] >= threshold]), len(dataset[dataset[:, feature_index] >= threshold])
        e_1 = (n_left/n) * e_1_left + (n_right/n) * e_1_right
        return e_0 - e_1
    
    def most_common_label(self, dataset) -> int:
        """
        Find the most common label in a dataset.
        Parameters:
        -----------
        dataset : numpy.ndarray
            The dataset for which to find the most common label.
        Returns:
        --------
        int
            The most common label in the dataset.
            
        """
        y = dataset[:, -1]
        return np.bincount(y).argmax()
    
    def accuracy_score(self, X, y) -> float:
        """
        Calculate the accuracy of the classifier on the given test data.
        Parameters:
        -----------
        X : numpy.ndarray
            The test input samples. Shape (n_samples, n_features).
        y : numpy.ndarray
            The true target values. Shape (n_samples,).
        Returns:
        --------
        float
            The accuracy of the classifier on the test data.
        """
        
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy

    
    


class TreeRegressor(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=0):
        super().__init__(max_depth, min_samples_split, min_gain)
        self.impurity_calculation = self.variance_reduction
        self.leaf_value_calculation = self.mean_target
        self.score_calculation = self.r_squared_score
        
    def variance_reduction(self, parent, left_child, right_child):
       """
       Compute the variance reduction of a split.
       Parameters:
       -----------
       parent : numpy.ndarray
           The parent dataset.
       left_child : numpy.ndarray
           The left child dataset.
       right_child : numpy.ndarray
           The right child dataset.
       Returns:
       --------
       float
           The variance reduction of the split.
       """
       weight_l = len(left_child) / len(parent)
       weight_r = len(right_child) / len(parent)
       return self.variance(parent) - (weight_l * self.variance(left_child) + weight_r * self.variance(right_child))
   
    def variance(self, dataset):
        """
        Compute the variance of a dataset.
        Parameters:
        -----------
        dataset : numpy.ndarray
            The dataset for which to compute the variance.
        Returns:
        --------
        float
            The variance of the dataset.
        """
        y = dataset[:, -1]
        mean = np.mean(y)
        return np.sum((y - mean) ** 2)
    
    def mean_target(self, dataset) -> float:
        """
        Compute the mean target value of a dataset.
        Parameters:
        -----------
        dataset : numpy.ndarray
            The dataset for which to compute the mean target value.
        Returns:
        --------
        float
            The mean target value of the dataset.
        """
        y = dataset[:, -1]
        return np.mean(y)
    
    def r_squared_score(self, X, y) -> float:
        """
        Calculate the R^2 score of the regressor on the given test data.
        Parameters:
        -----------
        X : numpy.ndarray
            The test input samples. Shape (n_samples, n_features).
        y : numpy.ndarray
            The true target values. Shape (n_samples,).
        Returns:
        --------
        float
            The R^2 score of the regressor on the test data.
        """
        y_pred = self.predict(X)
        mean = np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - mean) ** 2)
        return 1 - ss_res / ss_tot