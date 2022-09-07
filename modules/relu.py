import numpy as np

class ReLU():        
    def forward(self, x):
        self.x = x
        return np.maximum(0., x)

    def backward(self, grad):
        grad = grad * np.where(self.x <= 0, 0, 1).astype(self.x.dtype)
        return grad