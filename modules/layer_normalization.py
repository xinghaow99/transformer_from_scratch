import numpy as np

class LayerNormalization():
    # 2-dimension
    def __init__(self, optimizer, normalized_shape, eps=1e-05, data_type=np.float32):
        self.layer_name = "layernorm"
        self.optimizer = optimizer
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.data_type = data_type
        self.gamma = None
        self.beta = None
        self.init_weights()

    def init_weights(self):
        self.gamma = np.ones(self.normalized_shape).astype(self.data_type)
        self.beta = np.zeros(self.normalized_shape).astype(self.data_type)

    def register(self):
        self.layer_id = self.optimizer.count_layers(self.layer_name) // 2
        self.register_name = "{}_{}".format(self.layer_name, self.layer_id)
        self.optimizer.registered_layer_params["{}.gamma".format(self.register_name)] = {}
        self.optimizer.registered_layer_params["{}.beta".format(self.register_name)] = {}


    def forward(self, x):
        self.x = x
        x_T = x.T
        self.normalized_axis = tuple(np.arange(self.x.ndim - self.gamma.ndim).tolist())
        self.feature_size = self.gamma.size
        self.mean = np.mean(x_T, axis = 0)
        self.var = np.var(x_T,axis = 0)
        self.X_centered = (x_T - self.mean)
        self.stddev_inv = 1 / np.sqrt(self.var + self.eps)

        self.X_hat_T = self.X_centered * self.stddev_inv
        self.X_hat = self.X_hat_T.T
        
        self.output_data = self.gamma * self.X_hat + self.beta

        return self.output_data

    def backward(self, grad):
        self.grad_gamma = np.sum(grad * self.X_hat, axis = self.normalized_axis)
        self.grad_beta = np.sum(grad, axis = self.normalized_axis)
        grad_T = grad.T
        grad = (1 / self.feature_size) * np.expand_dims(self.gamma, axis = self.normalized_axis).T * self.stddev_inv * (
            self.feature_size * grad_T
            - np.sum(grad_T, axis = 0)
            - self.X_centered * np.power(self.stddev_inv, 2) * np.sum(error_T * self.X_centered, axis = 0)
            )
        grad = grad.T
        return grad

    def update_weights(self):
        self.optimizer.update(self.gamma, self.grad_gamma, "{}.gamma".format(self.register_name))
        self.optimizer.update(self.beta, self.grad_beta, "{}.beta".format(self.register_name))

