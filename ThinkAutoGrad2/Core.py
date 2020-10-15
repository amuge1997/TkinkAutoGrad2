from .Tensor import Tensor
from .Model import Model
from .Backward import backward


class Activate:
    from .Activate import Sigmoid, Relu
    Sigmoid = Sigmoid
    Relu = Relu


class Optimizer:
    from .Optimizer import Adam
    Adam = Adam


class Layers:
    from .Layers import Conv2d, Flatten, UpSample2d
    Conv2d = Conv2d
    Flatten = Flatten
    UpSample2d = UpSample2d


class Utils:
    from .Utils import Concat
    Concat = Concat


class Losses:
    from .Losses import MSE
    MSE = MSE













