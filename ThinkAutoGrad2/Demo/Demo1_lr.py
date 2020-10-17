
from ThinkAutoGrad2.Tensor import Tensor
from ThinkAutoGrad2.Activate import Sigmoid
import numpy as n


inps = Tensor(n.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
]))
labs = Tensor(n.array([
    [1],
    [1],
    [0],
    [0]
]))

w = Tensor(n.random.randn(2, 1), is_grad=True)
b = Tensor(n.zeros((1,)), is_grad=True)
c = Tensor(n.array(1 / 4))

z = inps @ w + b
loss = c * (labs - Sigmoid(z)()) * (labs - Sigmoid(z)())
g = n.ones(loss.shape, dtype=n.float32)

for i in range(5000):
    z = inps @ w + b
    outs = Sigmoid(z)()
    loss = c * (labs - outs) * (labs - outs)
    loss.backward(g)

    w.arr -= 1e-1 * w.grad
    b.arr -= 1e-1 * b.grad

    w.grad_zeros()
    b.grad_zeros()

    print('loss - {}'.format(n.sum(loss.arr)))

z = inps @ w + b
outs = Sigmoid(z)()

print('outs')
print(outs)















