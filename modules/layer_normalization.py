import cupy as cp

class LayerNormalization():
    # 2-dimension
    def __init__(self, optimizer, normalized_shape, eps=1e-05, data_type=cp.float32):
        self.layer_name = "layernorm"
        self.optimizer = optimizer
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.data_type = data_type
        self.gamma = None
        self.beta = None
        self.init_weights()
        self.register()

    def init_weights(self):
        self.gamma = cp.ones(self.normalized_shape).astype(self.data_type)
        self.beta = cp.zeros(self.normalized_shape).astype(self.data_type)

    def register(self):
        self.layer_id = self.optimizer.count_layers(self.layer_name) // 2
        self.register_name = "{}_{}".format(self.layer_name, self.layer_id)
        self.optimizer.register_params("{}.gamma".format(self.register_name), self.gamma)
        self.optimizer.register_params("{}.beta".format(self.register_name), self.beta)

    def forward(self, x):
        self.x = x
        x_mean = x.mean(axis=-1, keepdims=True)
        x_var = x.var(axis=-1, keepdims=True)
        lnorm = (x - x_mean) / cp.sqrt(x_var + self.eps)
        y = self.gamma * lnorm + self.beta
        return y

    def backward(self, grad):
        x = self.x
        x_mean = x.mean(axis=-1, keepdims=True)
        x_var = x.var(axis=-1, keepdims=True)
        lnorm = (x - x_mean) / cp.sqrt(x_var + self.eps)
        batch_size, seq_len, d = x.shape
        self.grad_gamma = grad.sum(axis=tuple(range(grad.ndim - 1)))
        self.grad_beta = cp.sum(grad * lnorm, axis=tuple(range(grad.ndim - 1)))
        grad_lnorm = grad * self.gamma
        grad_x = (
            d * grad_lnorm
            - grad_lnorm.sum(axis=-1, keepdims=True)
            - lnorm * (grad_lnorm * lnorm).sum(axis=-1, keepdims=True)
            ) / (d * cp.sqrt(x_var + self.eps))
        return grad_x

    def release_memory(self):
        del self.grad_gamma, self.grad_beta

    def update_weights(self):
        self.gamma = self.optimizer.update(self.gamma, self.grad_gamma, "{}.gamma".format(self.register_name))
        self.beta = self.optimizer.update(self.beta, self.grad_beta, "{}.beta".format(self.register_name))
        self.release_memory()