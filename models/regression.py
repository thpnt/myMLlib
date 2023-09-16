from typing import Any
import numpy as np
import matplotlib.pyplot as plt

class l1_regularization():
    """ 
    Regularization for lasso regression
    
    Add sum of absolute values of weights as a penalty term to the loss function.
    It shrinks less important features to 0.
    
    """
    
    def __init__(self, alpha) :
        self.alpha = alpha
        
    def __call__(self, w):
        # alpha coeficient * sum of absolute weights values
        return self.alpha * np.linalg.norm(w)
    
    def grad(self, w) :
        self.alpha * np.sign(w)
        
class l2_regularization():
    """ 
    Regularization for Ridge regression
    
    Add squared magnitude to loss function to regulate weights.
    
    """
    
    def __init__(self, alpha) :
        self.alpha = alpha
        
    def __call__(self, w):
        # alpha coeficient * sum of squared weights values
        return self.alpha * 0.5 * w.T.dot(w)
    
    def grad(self, w) :
        self.alpha * w

class l1_l2_regularization():
    """ 
    Regularization for Elastic Net Regression : combines l1 and l2
    
    """
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w) 
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr)



class Regression(object):
    def __init__(self, n_iterations, learning_rate) -> None:
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
    
    def initialize_weights(self, n_features) :
        """ Initialize random weights """
        limit = 1 / n_features**2
        self.w = np.random.uniform(-limit, limit, (n_features, ))
    
    def fit(self, X, y) :
        """ Mathematical explanations :
        X.shape = (n, m)
        y.shape = (n, )
        w.shape = (m, )
        
        we want to find the weights that minimize the MSE function.
        
        Y_pred = X . w

        MSE is defined by :
            mse = 1/n * sum((y - y_pred)^2)
            mse = np.mean(y - y_pred)
            
        to minimize the MSE loss funtion, we compute the gradient (vector of partial derivatives
        in respect to each weight)
        
            dMSE/dw = -2/n * sum(x * (y - y_pred))
            dMSE/dw = -2/n * X.T . (Y - Y_pred)
        or in numpy :
            dw = -2/n * np.dot(X.T, y - y_pred)
            
        Then we can update the weights : they move in the opposite direction of the gradient
            w = w - L * dw
        
        """
        #initialize bias with constant ones and weights
        X = np.insert(X, 0, 1, axis=1)
        y = y.flatten()
        self.initialize_weights(X.shape[1])
        #Compute MSE and store evolutions in training_errors for viz purposes
        self.traning_errors = []
    

        for _ in range(self.n_iterations) :
            #prediction with current weight and mse
            y_pred = np.dot(X, self.w)
            res = y - y_pred
            mse = np.mean((y-y_pred)**2) + self.regularization(self.w)
            self.traning_errors.append(mse)
            
            #Gradient descent
            dw = -2/X.shape[0] * np.dot(X.T, (y-y_pred)) + self.regularization.grad(self.w)
            self.w -= self.learning_rate * dw
            
    def weights(self) :
        return self.w
    
    def predict(self, X) :
        X = np.insert(X, 0, 1, axis=-1)
        y_pred = np.dot(X, self.w) 
        return y_pred
    
    def plot_mse(self) :
        plt.plot(list(range(len(self.traning_errors))), self.traning_errors, marker='o')  # 'o' adds markers to data points
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE over time')
        plt.grid(True)
        plt.show()
        
   
   
   
class LinearRegression(Regression) :
    """
    A class for Linear Regression modeling, which aims to predict a target variable (y) as a linear combination of independent features (X).
    
    Parameters:
    
    - n_iterations (int): The number of iterations in gradient descent. This parameter determines how many times the algorithm updates the model parameters during training. 
    
    - learning_rate (float): The step size for each parameter update in gradient descent. It controls the size of the steps taken towards minimizing the loss function. 
    A smaller value ensures stability but may slow down convergence, while a larger value may result in overshooting the optimal solution. 
    """ 
    def __init__(self, n_iterations=100, learning_rate=0.001):
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                            learning_rate=learning_rate)
    
    def fit(self, X, y) :
        super().fit(X, y)
    
    def weights(self) :
        super().weights()
    
    def predict(self, X) :
        super().predict(X)
    
    def plot_mse(self) :
        super().plot_mse()
        
        

class LassoRegression(Regression) :
    """
    Lasso Regression with L1 regularization.

    Parameters:
    
    - reg_factor (float): Strength of L1 regularization.

    - n_iterations (int): Number of gradient descent iterations.
    
    - learning_rate (float): Step size for parameter updates.
    """   
    def __init__(self, reg_factor, n_iterations=3000, learning_rate=0.01):
        self.regularization = l1_regularization(alpha=reg_factor)
        super().__init__(n_iterations, learning_rate)
    
    def fit(self, X, y):
        super().fit(X, y)
    
    def weights(self) :
        super().weights()
    
    def predict(self, X) :
        super().predict(X)
    
    def plot_mse(self) :
        super().plot_mse()
        
class RidgeRegression(Regression) :
    """
    Ridge Regression with L2 regularization.

    Parameters:
    
    - reg_factor (float): Strength of L2 regularization.

    - n_iterations (int): Number of gradient descent iterations.
    
    - learning_rate (float): Step size for parameter updates.
    """   
    def __init__(self, reg_factor, n_iterations=3000, learning_rate=0.01):
        self.regularization = l2_regularization(alpha=reg_factor)
        super().__init__(n_iterations, learning_rate)
    
    def fit(self, X, y):
        super().fit(X, y)
    
    def weights(self) :
        super().weights()
    
    def predict(self, X) :
        super().predict(X)
    
    def plot_mse(self) :
        super().plot_mse()
        
class ElasticNetRegression(Regression) :
    """
    ElasticNet Regression with L1 regularization.

    Parameters:
    - l1_ratio : weight on regulatization
    
    - reg_factor (float): Strength of L1-L2 regularization.

    - n_iterations (int): Number of gradient descent iterations.
    
    - learning_rate (float): Step size for parameter updates.
    """   
    def __init__(self, reg_factor, l1_ratio=0.5, n_iterations=3000, learning_rate=0.01):
        self.regularization = l1_l2_regularization(alpha=reg_factor, l1_ratio=l1_ratio)
        super().__init__(n_iterations, learning_rate)
    
    def fit(self, X, y):
        super().fit(X, y)
    
    def weights(self) :
        super().weights()
    
    def predict(self, X) :
        super().predict(X)
    
    def plot_mse(self) :
        super().plot_mse()
