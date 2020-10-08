from .Tensor import Tensor
import numpy as n
from .Check import grad_outs_check


# 1级测试
class Concat:
    def __init__(self, xls, axis):
        self.xls = xls
        self.axis = axis

    def __call__(self):
        z = Tensor(n.concatenate([i.arr for i in self.xls]), self, self.xls)
        return z

    @grad_outs_check
    def backward(self, grad):
        shape_ls = [i.arr.shape[self.axis] for i in self.xls]
        shape_len = len(self.xls[0].arr.shape)

        shape_new = 0
        gz = []
        head = [slice(None, None, None) for i in range(self.axis)]
        tail = [slice(None, None, None) for i in range(shape_len - self.axis - 1)]
        for s in shape_ls:
            shape_last = shape_new
            shape_new = shape_new + s
            this = [slice(shape_last, shape_new, )]
            slice_ = tuple(head + this + tail)
            gz.append(grad[slice_])
        return tuple(gz)


# 0级测试
class Flatten:
    def __init__(self, x):
        self.x = x
        self.x_shape = x.arr.shape

    def __call__(self):
        import functools
        z = self.x.reshape((self.x_shape[0], functools.reduce(lambda x, y: x * y, self.x_shape[1:])))
        return z


if __name__ == '__main__':
    a = Tensor(n.ones((1, 2)))
    b = Tensor(n.ones((1, 2)))

    c = Concat([a, b], axis=0)()

    g = n.ones_like(c.arr, dtype=n.float32)

    print(g.shape)
    c.backward(g)
    print(b.grad)

    print()










