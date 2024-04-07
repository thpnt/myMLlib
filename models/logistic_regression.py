import numpy
import math


# Logistic regression model
class LogisticRegression():
    """
    Logistic regression classifier.

    Parameters:
    -----------
    Learning_rate : float
        Step size for the parameter updates during gradient descent.


    Mathematical formulation:
    -------------------------
    The probability that a given input (x) belongs to class 1 is given by:
    P(Y=1|X) = sigmoid(w*x + b)
    The probability that a given input (x) belongs to class 0 is given by:
    P(Y=0|X) = 1 - P(Y=1|X)
    
    The likelihood of the data is given by:
    L = Product(P(Y=y[i]|X=x[i]))
    """    
    
    # Constructor
    def __init__(self, learning_rate=0.1, n_iterations=4000):
        self.learning_rate = learning_rate
        
    
    # Sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))
    
    # Log-likelihood
    def log_likelihood(self, y, y_pred):
        return -y * math.log(y_pred) - (1 - y) * math.log(1 - y_pred)
    
    # Initialize parameters randomly
    def initialize_parameters(self, X):
        self.w = numpy.random.uniform(-1, 1, X.shape[1])
        self.b = 0
    
    # Fit the model
    def fit(self, X, y):
        self.initialize_parameters(X)
        
        for i in range(self.n_iterations):
            # Calculate the probability of the data belonging to class 1
            y_pred = self.sigmoid(numpy.dot(X, self.w) + self.b)
            
            # Calculate the gradient
            w_grad = numpy.dot(X.T, (y_pred - y))
            b_grad = numpy.sum(y_pred - y)
            
            # Update the parameters
            self.w -= self.learning_rate * w_grad
            self.b -= self.learning_rate * b_grad
            
            # Print the log-likelihood every 100 iterations
            if i % 100 == 0:
                print(self.log_likelihood(y, y_pred).mean())
        
    
    # Predict the class labels
    def predict(self, X):
        y_pred = numpy.round(self.sigmoid(numpy.dot(X, self.w) + self.b))
        return y_pred
