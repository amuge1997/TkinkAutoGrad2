from .Tensor import Tensor
from .Model import Model
from .Backward import backward


class Activate:
    from .Activate import Relu, Sigmoid
    Relu = Relu
    Sigmoid = Sigmoid


class Layers:
    from .Layers import Conv2d, Flatten, UpSample2d
    Conv2d = Conv2d
    Flatten = Flatten
    UpSample2d = UpSample2d


class Losses:
    from .Losses import MSE
    MSE = MSE


class RNN:
    from .RNN import RNN
    RNN = RNN


class Optimizer:
    from .Optimizer import Adam
    Adam = Adam


class Utils:
    from .Utils import Concat
    Concat = Concat
















