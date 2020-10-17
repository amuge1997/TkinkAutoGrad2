from .Activate import Tanh
from .Utils import Concat


class RecurrentNN:
    pass


class RNN(RecurrentNN):
    def __init__(self, x, h, u, w, v, b):
        # x.shape = batch, time_step, in_dims
        # h.shape = batch, hi_dims
        # u.shape = in_dims, hi_dims
        # w.shape = hi_dims, hi_dims
        # v.shape = hi_dims, ou_dims
        # b.shape = hi_dims
        self.x = x
        self.h = h
        self.u = u
        self.w = w
        self.v = v
        self.b = b

    def __call__(self):
        batch, time_step = self.x.shape[0], self.x.shape[1]
        last_h = []
        for bah in range(batch):
            tsp_h = self.h[bah:bah+1, :]
            for tsp in range(time_step):
                p1 = self.x[bah:bah+1, tsp:tsp+1, :] @ self.u
                p2 = tsp_h @ self.w
                p3 = p1 + p2 + self.b
                tsp_h = Tanh(p3)()
            last_h.append(tsp_h)
        last_h = Concat(last_h, 0)()
        z = last_h @ self.v
        return z, last_h


class LSTM(RecurrentNN):
    pass













