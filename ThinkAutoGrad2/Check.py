
# 输出检测


def grad_outs_check(func):
    def warp(self, grad):
        ret = func(self, grad)
        if not isinstance(ret, tuple):
            raise Exception('返回的梯度必须是元组形式')
        return ret
    return warp










