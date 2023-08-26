import numpy as np
from regression import Regression


np.random.seed(42)     
m = 100
n = 5
X = np.random.rand(m, n)
#print(f'X shape : {X.shape}')

true_weights = np.random.rand(n, 1)

true_bias = 2.5
y = np.dot(X, true_weights) + true_bias + np.random.randn(m, 1) * 0.1




X_pred = np.random.rand(1, n)

regressor = Regression(n_iterations=10000, learning_rate=0.0001)

print(f'fitting in progress')
regressor.fit(X, y)
print(f'fitting completed.')

regressor.predict(X_pred)
print(f'prediction finished')

print(f'weight of model are :')
print(regressor.weights())

regressor.plot_mse()