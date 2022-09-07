from re import L
import numpy as np

class Adam():
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.98, eps=1e-9, warmup_steps=4000):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step = 0
        self.warmup_steps = warmup_steps
        self.registered_layer_params = {}

    def step(self):
        self.step += 1

    def set_lr(self):
        self.lr = 512 ** -0.5 * min((self.step+1) ** -0.5, (self.step+1) * self.warmup_steps ** -1.5)

    def save_params(self, layer_name, t, mean, var):
        self.registered_layer_params[layer_name] = {
                "t": t,
                "mean": mean,
                "var": var
            }

    def count_layers(self, layer_name):
        cnt = 0
        for key in self.registered_layer_params.keys():
            if key.startswith(layer_name):
                cnt += 1
        return cnt

    def update(self, param, param_grad, layer_name):
        if layer_name not in self.registered_layer_params:
            self.save_params(layer_name=layer_name, t=0, mean=np.zeros_like(param_grad), var=np.zeros_like(param_grad))
        t = self.registered_layer_params[layer_name]["t"]
        mean = self.registered_layer_params[layer_name]["mean"]
        var = self.registered_layer_params[layer_name]["var"]
        # update
        t += 1
        mean = self.beta1 * mean + (1 - self.beta1) * param_grad
        var = self.beta2 * var + (1 - self.beta2) * param_grad ** 2
        self.save_params(layer_name, t, mean, var)
        mean_hat = mean / (1 - self.beta1 ** t)
        var_hat = var / (1 - self.beta2 ** t)
        self.set_lr()
        param = param - self.lr * mean_hat / (np.sqrt(var_hat) + self.eps)
        return param
