import cupy as cp

class Softmax():
    def __init__(self):
        self.axis = -1

    def forward(self, x):
        e_x = cp.exp(x - cp.max(x, axis = self.axis, keepdims=True))
        self.y =  e_x / cp.sum(e_x, axis = self.axis, keepdims=True)
        del e_x
        return cp.nan_to_num(self.y, nan=0.)

    def backward(self, grad_y):
        # https://sgugger.github.io/a-simple-neural-net-in-numpy.html
        grad_x = self.y * (grad_y - (grad_y * self.y).sum(axis=self.axis, keepdims=True))
        # print('aa')
        # print(grad_x)
        # shape = self.y.shape
        # grad_y = grad_y.reshape(-1, grad_y.shape[-1])
        # self.y = self.y.reshape(-1, self.y.shape[-1])
        # a = cp.eye(self.y.shape[-1])
        # temp1 = cp.zeros((self.y.shape[0], self.y.shape[1], self.y.shape[1]),dtype=cp.float32)
        # temp2 = cp.zeros((self.y.shape[0], self.y.shape[1], self.y.shape[1]),dtype=cp.float32)
        # temp1 = cp.einsum('ij,jk->ijk',self.y,a)
        # temp2 = cp.einsum('ij,ik->ijk',self.y,self.y)
        # joc = (temp1 - temp2).reshape(tuple(list(shape) + [self.y.shape[-1]]))
        # grad_x = grad_y @ joc
        # print(grad_x)
        # print(grad_x.shape)
        # self.y = self.y.reshape(-1, self.y.shape[-1])
        # grad_y = grad_y.reshape(-1, grad_y.shape[-1])
        # grad_x = (self.y * (grad_y - (grad_y * self.y).sum(axis=self.axis, keepdims=True))).reshape(shape)
        # print('bb')
        # print(grad_x)
        del self.y
        return cp.nan_to_num(grad_x, nan=0.)

