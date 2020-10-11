import numpy as n
from .Tensor import Tensor


class Loss:
    def __init__(self):
        pass


class MSE(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, prds, labs):
        c = Tensor(n.array([1 / prds.shape[0], ]))
        loss = c * (labs - prds) * (labs - prds)
        return loss
















