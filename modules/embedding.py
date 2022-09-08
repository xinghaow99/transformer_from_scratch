import numpy as np
class Embedding():
    def __init__(self, num_embeddings, embedding_dim, optimizer, data_type=np.float32):
        self.layer_name = "embedding"
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.optimizer = optimizer
        self.data_type = data_type
        self.weights = None
        self.init_weights()
        self.register()

    def init_weights(self):
        self.weights = np.random.normal(0, 1, (self.num_embeddings, self.embedding_dim)).astype(self.data_type)

    def register(self):
        weights_registered_name = '{}_{}'.format(self.layer_name, 'weights')
        cnt= self.optimizer.count_layers(weights_registered_name)
        self.weights_registered_name = "{}_{}".format(weights_registered_name, cnt)
        self.optimizer.register_params(self.weights_registered_name, self.weights)

    def forward(self, indicies):
        self.indicies = indicies
        self.output = np.take(self.weights, self.indicies, axis=0)
        return self.output
    
    def backward(self, grad_y):
        self.grad_weights = np.sum(grad_y)
        return None

    def update_weights(self):
        self.optimizer.update(self.weights, self.grad_weights, self.weights_registered_name)
