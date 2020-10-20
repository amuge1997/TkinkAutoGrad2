import numpy as n
from .Tensor import Tensor


def check_losses_inps_shape_equal(func):
    def wrap(self, prds, labs):
        prds_shape = tuple(prds.shape)
        labs_shape = tuple(labs.shape)
        if prds_shape != labs_shape:
            raise Exception('prds与labs必须相同: prds.shape={} labs.shape={}'.format(prds_shape, labs_shape))
        ret = func(self, prds, labs)
        return ret
    return wrap


class MSE:
    def __init__(self):
        super().__init__()

    @check_losses_inps_shape_equal
    def __call__(self, prds, labs):
        c = Tensor(n.array([1 / prds.shape[0], ]))
        loss = c * (labs - prds) * (labs - prds)
        return loss
















