import numpy as np
class Embedding():
    def __init__(self, num_embeddings, embedding_dim, data_type=np.float32):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.data_type = data_type
        self.weights = None
        self.init_weights()

    def init_weights(self):
        self.weights = np.random.normal(0, 1, (self.num_embeddings, self.embedding_dim)).astype(self.data_type)

    def forward(self, indicies):
        self.indicies = indicies
        self.output = np.take(self.weights, input, axis=0)
        return self.output
    
    def backward(self, grad_y):
        self.grad_weights = np.sum(grad_y)
        return None


embedding = Embedding(10, 3)
input = np.array([[1,2,3,4], [5,6,7,8]])
print(embedding.forward(input))