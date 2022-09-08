import numpy as np

class CrossEntropy():
    def __init__(self, vocab_size):
        self.eps = 1e-6
        self.vocab_size = vocab_size

    def one_hot(self, label):
        batch_size, seq_len = label.shape
        batch_one_hot = []
        for batch in range(batch_size):
            one_hot = np.zeros((seq_len, self.vocab_size))
            one_hot[np.arange(seq_len), label[batch]] = 1
            batch_one_hot.append(one_hot)
        self.one_hot_label = np.asarray(batch_one_hot)
        return self.one_hot_label

    def forward(self, pred, label):
        label = self.one_hot(label)
        return -np.sum(label * np.log(pred + self.eps))

    def grad(self, pred, label):
        return pred - self.one_hot_label