from .Activate import Tanh, Sigmoid
from .Utils import Concat
from .Check import grad_outs_check
from .Tensor import Tensor


class RNN:
    def __init__(self, x, h, u, w, b):
        # x.shape = batch, time_step, in_dims
        # h.shape = batch, hi_dims
        # u.shape = in_dims, hi_dims
        # w.shape = hi_dims, hi_dims
        # b.shape = hi_dims
        self.x = x
        self.h = h
        self.u = u
        self.w = w
        self.b = b

    def __call__(self):
        batch, time_step = self.x.shape[0], self.x.shape[1]
        tsp_h = self.h[:, :]
        for tsp in range(time_step):
            p1 = self.x[:, tsp:tsp+1, :] @ self.u
            p2 = tsp_h @ self.w
            p3 = p1 + p2 + self.b
            tsp_h = Tanh(p3)()
        z = Concat(tsp_h, 0)()
        return z


# 1级测试
class LSTMCell:
    def __init__(self, x, hc, wf, bf, wi, bi, wc, bc, wo, bo):
        # x.shape = batch, in_dims
        # hc.shape = batch, hi_dims + ci_dims
        # hi_dims = ci_dims
        self.x = x
        self.hc = hc
        # 遗忘门
        # wf.shape = hi_dims+in_dims, ci_dims
        # bf.shape = ci_dims
        self.wf = wf
        self.bf = bf
        # 输入门
        # wi.shape = hi_dims+in_dims, ci_dims
        # bi.shape = ci_dims
        self.wi = wi
        self.bi = bi
        # wc.shape = hi_dims+in_dims, ci_dims
        # bc.shape = ci_dims
        self.wc = wc
        self.bc = bc
        # 输出门
        # wo.shape = hi_dims+in_dims, ci_dims
        # bo.shape = ci_dims
        self.wo = wo
        self.bo = bo

        # 梯度隔离

        # 输出隔离
        self.ht_ct = None
        # 输入隔离
        self.x_copy = self.x.copy()
        self.hc_copy = self.hc.copy()
        self.wf_copy = self.wf.copy()
        self.bf_copy = self.bf.copy()
        self.wi_copy = self.wi.copy()
        self.bi_copy = self.bi.copy()
        self.wc_copy = self.wc.copy()
        self.bc_copy = self.bc.copy()
        self.wo_copy = self.wo.copy()
        self.bo_copy = self.bo.copy()

    def __call__(self):
        x = self.x_copy
        hc = self.hc_copy
        wf = self.wf_copy
        bf = self.bf_copy
        wi = self.wi_copy
        bi = self.bi_copy
        wc = self.wc_copy
        bc = self.bc_copy
        wo = self.wo_copy
        bo = self.bo_copy

        half_size = int(hc.shape[1] / 2)
        h, c = hc[:, :half_size], hc[:, half_size:]
        hx = Concat([h, x], 1)()
        # 遗忘门
        ft = Sigmoid(hx @ wf + bf)()    # 遗忘比例
        ct = c * ft
        # 输入门
        it = Sigmoid(hx @ wi + bi)()    # 记忆比例
        ct_ = Tanh(hx @ wc + bc)()      # 信息
        ct = ct + it * ct_
        # 输出门
        ot = Sigmoid(hx @ wo + bo)()
        ht = ot * Tanh(ct)()
        self.ht_ct = Concat([ht, ct], 1)()
        z = Tensor(self.ht_ct.arr, self, (self.x,  self.hc,
                                          self.wf, self.bf,
                                          self.wi, self.bi,
                                          self.wc, self.bc,
                                          self.wo, self.bo))
        return z

    @grad_outs_check
    def backward(self, grad):
        self.ht_ct.backward(grad)
        gz = (
            self.x_copy.grad,  self.hc_copy.grad,
            self.wf_copy.grad, self.bf_copy.grad,
            self.wi_copy.grad, self.bi_copy.grad,
            self.wc_copy.grad, self.bc_copy.grad,
            self.wo_copy.grad, self.bo_copy.grad
        )
        return gz


# 1级测试
class LSTM:
    def __init__(self, x, h, c, wf, bf, wi, bi, wc, bc, wo, bo):
        # x.shape = batch, time_step, in_dims
        # h.shape = batch, hi_dims
        # c.shape = batch, ci_dims
        # hi_dims = ci_dims
        self.x = x
        self.h = h
        self.c = c
        self.hc = Concat([h, c], axis=1)()
        # 遗忘门
        # wf.shape = hi_dims+in_dims, ci_dims
        # bf.shape = ci_dims
        self.wf = wf
        self.bf = bf
        # 输入门
        # wi.shape = hi_dims+in_dims, ci_dims
        # bi.shape = ci_dims
        self.wi = wi
        self.bi = bi
        # wc.shape = hi_dims+in_dims, ci_dims
        # bc.shape = ci_dims
        self.wc = wc
        self.bc = bc
        # 输出门
        # wo.shape = hi_dims+in_dims, ci_dims
        # bo.shape = ci_dims
        self.wo = wo
        self.bo = bo

    def __call__(self):
        batch, time_step, in_dims = self.x.shape

        tsp_hc = self.hc
        for tsp in range(time_step):
            tsp_x = self.x[:, tsp, :]  # 维度降低,去除time_step维度
            tsp_hc = LSTMCell(tsp_x, tsp_hc, self.wf, self.bf, self.wi, self.bi, self.wc, self.bc, self.wo, self.bo)()

        half_size = int(self.hc.shape[1] / 2)
        last_h = tsp_hc[:, :half_size]
        last_c = tsp_hc[:, half_size:]
        return last_h, last_c


# 效率极低
# lstm递归函数调用复杂度至少为O(2^n),指数级
# rnn递归函数调用复杂度为O(n),常数级
class LSTMUnable:
    def __init__(self, x, h, c, wf, bf, wi, bi, wc, bc, wo, bo):
        # x.shape = batch, time_step, in_dims
        # h.shape = batch, hi_dims
        # c.shape = batch, ci_dims
        # hi_dims = ci_dims
        self.x = x
        self.h = h
        self.c = c
        # 遗忘门
        # wf.shape = hi_dims+in_dims, ci_dims
        # bf.shape = ci_dims
        self.wf = wf
        self.bf = bf
        # 输入门
        # wi.shape = hi_dims+in_dims, ci_dims
        # bi.shape = ci_dims
        self.wi = wi
        self.bi = bi
        # wc.shape = hi_dims+in_dims, ci_dims
        # bc.shape = ci_dims
        self.wc = wc
        self.bc = bc
        # 输出门
        # wo.shape = hi_dims+in_dims, ci_dims
        # bo.shape = ci_dims
        self.wo = wo
        self.bo = bo

    def __call__(self):
        batch, time_step = self.x.shape[0], self.x.shape[1]
        last_c = []
        last_h = []

        for bah in range(batch):
            tsp_h = self.h[bah:bah+1, :]
            tsp_c = self.c[bah:bah+1, :]

            for tsp in range(time_step):
                tsp_x = self.x[bah:bah + 1, tsp, :]     # 维度降低,去除time_step维度
                hx_cat = Concat([tsp_h, tsp_x], 1)()
                # 遗忘门
                ft = Sigmoid(hx_cat @ self.wf + self.bf)()
                ct = ft * tsp_c
                # 输入门
                it = Sigmoid(hx_cat @ self.wi + self.bi)()
                c_ = Tanh(hx_cat @ self.wc + self.bc)()
                ct = ct + it * c_
                # 输出门
                ot = Sigmoid(hx_cat @ self.wo + self.bo)()
                ht = ot * Tanh(ct)()

                tsp_h = ht
                tsp_c = ct
            last_h.append(tsp_h)
            last_c.append(tsp_c)
        ret_h = Concat(last_h, 0)()
        ret_c = Concat(last_c, 0)()
        return ret_h, ret_c












