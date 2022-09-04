import numpy as np

class LayerNormalization():
    # 2-dimension
    def __init__(self, embedding_dim, eps=1e-05, data_type=np.float32):
        self.layer_name = "layernorm"
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.data_type = data_type
        self.gamma = None
        self.beta = None
        self.init_weights()

    def init_weights(self):
        self.gamma = np.ones(self.embedding_dim).astype(self.data_type)
        self.beta = np.zeros(self.embedding_dim).astype(self.data_type)

    def register(self):
        self.layer_id = self.optimizer.count_layers(self.layer_name) // 2
        self.register_name = "{}_{}".format(self.layer_name, self.layer_id)
        self.optimizer.registered_layer_params["{}.gamma".format(self.register_name)] = {}
        self.optimizer.registered_layer_params["{}.beta".format(self.register_name)] = {}


    def forward(self, x):
        self.x = x
        batch_size = x.shape[0]
        x_mean = np.mean(x, axis=1).reshape(batch_size, 1)
        x_var = np.var(x, axis=1).reshape(batch_size, 1)
        a = x - x_mean
        b = np.sqrt(x_var+self.eps)
        self.a = a
        self.b = b
        return a / b

    def backward(self, grad_y):
        self.grad_gamma = np.sum(grad_y * self.a / self.b, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_y, axis=0, keepdims=True)
        dlxhat = grad_y * self.gamma
        dxhatx = 1/self.b
        dlvar = -0.5*np.sum(self.gamma*self.a*self.b**(-3)*grad_y,axis=1,keepdims=True)
        dlvarx = 2*self.a/self.embedding_dim
        dlmu = -1.*np.sum(dlxhat/self.b,axis=1,keepdims=True)-2.*np.sum(dlvar*self.a,axis=1,keepdims=True)/self.embedding_dim
        self.grad_x = dlxhat*dxhatx + dlvar*dlvarx + dlmu/self.embedding_dim

    def update_weights(self):
        self.optimizer.update(self.gamma, self.grad_gamma, "{}.gamma".format(self.register_name))
        self.optimizer.update(self.beta, self.grad_beta, "{}.beta".format(self.register_name))

