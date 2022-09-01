import numpy as np

class Linear():
    def __init__(self, in_features, out_features, use_bias=True, data_type=np.float32):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weights = None
        self.bias= None
        self.data_type = data_type
        self.init_weights()

    def init_weights(self):
        sqrt_k = 1. / np.sqrt(self.in_features)
        self.weights = np.random.uniform(-sqrt_k, sqrt_k, (self.in_features, self.out_features)).astype(self.data_type)
        self.bias = np.zeros(self.out_features).astype(self.data_type) if self.use_bias else np.random.uniform(-sqrt_k, sqrt_k, self.out_features)
    
    def forward(self, x):
        self.x = x
        self.output = x @ self.weights + self.bias
        return self.output
    
    def backward(self, grad_y):
        # https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        self.grad_weights = np.sum(self.x.T @ grad_y)
        self.grad_bias = np.sum(grad_y)
        self.grad_x = grad_y @ self.weights.T
        return self.grad_x
    
