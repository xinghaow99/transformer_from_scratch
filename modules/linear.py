from re import L
import numpy as np

class Linear():
    def __init__(self, in_features, out_features, optimizer, use_bias=True, data_type=np.float32):
        self.layer_name = "linear"
        self.in_features = in_features
        self.out_features = out_features
        self.optimizer = optimizer
        self.use_bias = use_bias
        self.weights = None
        self.bias= None
        self.data_type = data_type
        self.init_weights()
        self.register()


    def init_weights(self):
        sqrt_k = 1. / np.sqrt(self.in_features)
        self.weights = np.random.uniform(-sqrt_k, sqrt_k, (self.in_features, self.out_features)).astype(self.data_type)
        self.bias = np.zeros(self.out_features).astype(self.data_type) if self.use_bias else np.random.uniform(-sqrt_k, sqrt_k, self.out_features)
    
    def register(self):
        weights_registered_name = '{}_{}'.format(self.layer_name, 'weights')
        cnt= self.optimizer.count_layers(weights_registered_name)
        self.weights_registered_name = "{}_{}".format(weights_registered_name, cnt)
        self.optimizer.register_params(self.weights_registered_name, self.weights)
        if self.use_bias:
            bias_registered_name = '{}_{}'.format(self.layer_name, 'bias')
            cnt= self.optimizer.count_layers(bias_registered_name)
            self.bias_registered_name = "{}_{}".format(bias_registered_name, cnt)
            self.optimizer.register_params(self.bias_registered_name, self.bias)

    def forward(self, x):
        self.x = x
        self.output = x @ self.weights + self.bias
        return self.output
    
    def backward(self, grad):
        # https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        self.grad_weights = np.sum(self.x.transpose(0, 2, 1) @ grad, axis=0)
        self.grad_bias = np.sum(grad, axis=(0, 1))
        self.grad = np.dot(grad, self.weights.T)
        return self.grad
    
    def update_weights(self):
        self.optimizer.update(self.weights, self.grad_weights, self.weights_registered_name)
        if self.use_bias:
            self.optimizer.update(self.bias, self.grad_bias, self.bias_registered_name)
