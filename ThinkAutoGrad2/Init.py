from .Tensor import Tensor
import numpy as n


def check_xavier_inps(func):
    def wrap(shape, in_dims=None, out_dims=None, is_grad=False):
        if len(shape) == 1:
            if in_dims is None:
                in_dims = shape[0]
                out_dims = 0
            if out_dims is not None:
                raise Exception('out_dims错误')
        elif len(shape) == 2:
            if in_dims is None:
                in_dims = shape[0]
            if out_dims is None:
                out_dims = shape[1]
        return func(shape, in_dims, out_dims, is_grad)
    return wrap


class Init:

    # 1级
    @staticmethod
    @check_xavier_inps
    def xavier(shape, in_dims=None, out_dims=None, is_grad=False):
        mean = 0.
        std = n.sqrt(1 / (in_dims + out_dims))
        ret = Init.gaussian(shape, mean, std, is_grad)
        return ret

    # 1级
    @staticmethod
    def gaussian(shape, mean, std, is_grad=False):
        ret = Tensor(n.random.normal(mean, std, shape), is_grad=is_grad)
        return ret

    # 1级
    @staticmethod
    def zeros(shape, is_grad=False):
        ret = Tensor(n.zeros(shape), is_grad=is_grad)
        return ret

    # 0级
    @staticmethod
    def ones(shape, is_grad=False):
        ret = Tensor(n.ones(shape), is_grad=is_grad)
        return ret








