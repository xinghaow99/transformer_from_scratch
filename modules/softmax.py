import numpy as np

class Softmax():
    def __init__(self):
        self.axis = -1

    def forward(self, x):
        self.x = x
        e_x = np.exp(x - np.max(x, axis = self.axis, keepdims=True))
        self.y =  e_x / np.sum(e_x, axis = self.axis, keepdims=True)
        return self.y

    def backward(self, grad_y):
        # https://sgugger.github.io/a-simple-neural-net-in-numpy.html
        grad_x = self.y * (grad_y - (grad_y * self.y).sum(axis=self.axis, keepdims=True))
        return grad_x

