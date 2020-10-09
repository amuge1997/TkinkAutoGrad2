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


class UpSample2d:
    def __init__(self, x, stride):
        self.x = x
        self.x_shape = x.shape
        self.stride_hw = stride

    def __call__(self):
        inputs = self.x.arr
        n_samples, channels, height, width = self.x_shape
        stride_h, stride_w = self.stride_hw
        z = n.zeros((n_samples, channels, height * stride_h, width * stride_w))
        for hi in range(height):
            for wi in range(width):
                z[:, :, hi*stride_h:(hi+1)*stride_h, wi*stride_w:(wi+1)*stride_w] = \
                    n.ones((n_samples, channels, stride_h, stride_w)) * inputs[:, :, hi:hi+1, wi:wi+1]
        z = Tensor(z, self, (self.x, ))
        return z

    @grad_outs_check
    def backward(self, grad):
        n_samples, channels, grad_height, grad_width = grad.shape
        stride_h, stride_w = self.stride_hw
        if grad_height % stride_h != 0 or grad_width % stride_w != 0:
            raise Exception('error')
        out_height = int(grad_height / stride_h)
        out_width = int(grad_width / stride_w)
        gz = n.zeros((n_samples, channels, out_height, out_width))
        for hi in range(out_height):
            for wi in range(out_width):
                temp = n.sum(grad[:, :, hi*stride_h:(hi+1)*stride_h, wi*stride_w:(wi+1)*stride_w], axis=3, keepdims=True)
                temp = n.sum(temp, axis=2, keepdims=True)
                gz[:, :, hi:hi+1, wi:wi+1] = temp
        return (gz, )


if __name__ == '__main__':
    a = Tensor(n.ones((1, 2)))
    b = Tensor(n.ones((1, 2)))

    c = Concat([a, b], axis=0)()

    g = n.ones_like(c.arr, dtype=n.float32)

    print(g.shape)
    c.backward(g)
    print(b.grad)

    print()










