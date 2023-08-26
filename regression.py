import numpy as np
import matplotlib.pyplot as plt

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
            mse = np.mean((y-y_pred)**2)
            self.traning_errors.append(mse)
            
            #Gradient descent
            dw = -2/X.shape[0] * np.dot(X.T, (y-y_pred))
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
        
   