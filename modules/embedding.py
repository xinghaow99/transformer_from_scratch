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

    def init_weights(self):
        self.weights = np.random.normal(0, 1, (self.num_embeddings, self.embedding_dim)).astype(self.data_type)

    def register(self):
        self.layer_id = self.optimizer.count_layers(self.layer_name)
        self.register_name = "{}_{}".format(self.layer_name, self.layer_id)
        self.optimizer.registered_layer_params["{}.weights".format(self.register_name)] = {}

    def forward(self, indicies):
        self.indicies = indicies
        self.output = np.take(self.weights, input, axis=0)
        return self.output
    
    def backward(self, grad_y):
        self.grad_weights = np.sum(grad_y)
        return None

    def update_weights(self):
        self.optimizer.update(self.weights, self.grad_weights, "{}.weights".format(self.register_name))


embedding = Embedding(10, 3)
input = np.array([[1,2,3,4], [5,6,7,8]])
print(embedding.forward(input))