from .Tensor import Tensor
from .Check import grad_outs_check
import numpy as n


# 激活函数


# 0级测试
class Relu:
    def __init__(self, x):
        self.x = x
        self.x_shape = x.arr.shape

    def __call__(self):
        z = Tensor(n.where(self.x.arr > 0, self.x.arr, 0.), self, (self.x,))
        return z

    @grad_outs_check
    def backward(self, grad):
        gz = grad * n.where(self.x.arr > 0, 1.0, 0.)
        return (gz,)


# 3级测试
class Sigmoid:
    def __init__(self, x):
        self.x = x
        self.x_shape = x.arr.shape

    def __call__(self):
        z = Tensor(1 / (1 + n.exp(- self.x.arr)), self, (self.x,))
        return z

    @grad_outs_check
    def backward(self, grad):
        gz = grad * (1 / (1 + n.exp(- self.x.arr))) * (1 - 1 / (1 + n.exp(- self.x.arr)))
        return (gz,)







