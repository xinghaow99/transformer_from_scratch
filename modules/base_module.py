import numpy as np

class BaseModule(object):
    def __init__(self, training=True):
        self.training = training
    
    def forward(self, x):
        pass

    def backward(self, grad_y):
        pass