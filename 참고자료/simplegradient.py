import numpy as np

class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.001, epoch=10):
        self.learning_rate = learning_rate
        self.epoch = epoch
    
    def loss_function(self, X, y, a, b):
      data = np.column_stack((X,y))
      return sum((y-(a*x+b))**2 for x, y in data)/len(data)
    
    def fit(self, X, y):
        b = 0
        a = 0
        n = X.shape[0]
        for _ in range(self.epoch * n):
            if _%100 == 0:
              print('epoch:', _)
            print('loss=', self.loss_function(X, y, a, b), 'a=', a, 'b=', b)
            b_gradient = -2 * np.sum(y - (a*X + b)) / n
            a_gradient = -2 * np.sum(X*(y - (a*X + b))) / n
            b = b - (self.learning_rate * b_gradient)
            a = a - (self.learning_rate * a_gradient)
        self.a, self.b = a, b
    
    def partial_fit(self, X, y):
        b = self.b
        a = self.a
        n = X.shape[0]
        for _ in range(n):
            print('loss=', self.loss_function(X, y, a, b), 'a=', a, 'b=', b)
            b_gradient = -2 * np.sum(y - (a*X + b)) / n
            a_gradient = -2 * np.sum(X*(y - (a*X + b))) / n
            b = b - (self.learning_rate * b_gradient)
            a = a - (self.learning_rate * a_gradient)
        self.a, self.b = a, b

    def predict(self, X):
        return self.a*X + self.b