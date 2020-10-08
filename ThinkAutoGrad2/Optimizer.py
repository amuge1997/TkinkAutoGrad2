import numpy as n


class Adam:
    def __init__(self, lr, p1=0.9, p2=0.999):
        self.dc = dict()

        self.p1 = p1
        self.p2 = p2
        self.e = 1e-8
        self.lr = lr

    def run(self, w):
        # type(w) = Tensor
        w_id = id(w)
        dc = self.dc
        p1 = self.p1
        p2 = self.p2
        e = self.e
        lr = self.lr
        arr = w.arr
        grad = w.grad
        if w_id not in dc:
            epoch = 1
            dc[w_id] = {
                'epoch': 0,
                's': n.zeros(w.shape),
                'r': n.zeros(w.shape)
            }
        else:
            epoch = dc[w_id]['epoch']
        w_dc = dc[w_id]

        w_dc['epoch'] += 1
        w_dc['s'] = p1 * w_dc['s'] + (1 - p1) * grad
        w_dc['r'] = p2 * w_dc['r'] + (1 - p2) * grad ** 2

        s = w_dc['s'] / (1 - p1 ** epoch)
        r = w_dc['r'] / (1 - p2 ** epoch)

        arr += - lr * s / (n.sqrt(r) + e)














