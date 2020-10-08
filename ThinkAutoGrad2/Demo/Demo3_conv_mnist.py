from ThinkAutoGrad2.Utils_conv import Conv2d
from ThinkAutoGrad2.Utils import Flatten
from ThinkAutoGrad2.Tensor import Tensor
from ThinkAutoGrad2.Activate import Sigmoid, Relu
import numpy as n


def test1():
    in_c = 1
    out_c = 2
    image = Tensor(n.ones((1, in_c, 5, 5)))
    kernel = Tensor(n.ones((out_c, in_c, 3, 3)))
    bias = Tensor(n.ones(out_c))
    image2 = Conv2d(image, kernel, bias, (1, 1), is_padding=False)()

    grad = n.ones_like(image2.arr)
    image2.backward(grad)

    print(image2.shape)


def one_hot_encoding(labels, num_class=None):
    if num_class is None:
        num_class = n.max(labels) + 1
    one_hot_labels = n.zeros((len(labels), num_class))
    one_hot_labels[n.arange(len(labels)), labels] = 1
    return one_hot_labels.astype('int')


def load_data():
    import cv2 as cv
    from SB_MNIST import load_MNIST

    data_x, data_y = load_MNIST()

    x_ls = []
    for i in data_x:
        x = cv.resize(i, (10, 10))
        x = x[n.newaxis, ...]
        x_ls.append(x)
    data_x = n.concatenate(x_ls)

    data_x = data_x
    data_y = data_y

    data_y = one_hot_encoding(data_y, 10)

    return data_x, data_y


adam_dc = {}


def adam(w, grads, lr, epoch):
    p1 = 0.9
    p2 = 0.999
    e = 1e-8
    w_id = id(w)
    global adam_dc
    if w_id not in adam_dc:
        adam_dc[w_id] = {
            's': n.zeros_like(w),
            'r': n.zeros_like(w)
        }

    adam_dc[w_id]['s'] = p1 * adam_dc[w_id]['s'] + (1 - p1) * grads
    adam_dc[w_id]['r'] = p2 * adam_dc[w_id]['r'] + (1 - p2) * grads ** 2

    s = adam_dc[w_id]['s'] / (1 - p1 ** epoch)
    r = adam_dc[w_id]['r'] / (1 - p2 ** epoch)

    ret = - lr * s / (n.sqrt(r) + e)
    return ret


def test2():
    data_x, data_y = load_data()

    data_x = data_x[:, n.newaxis, ...]
    data_x = data_x / 255

    n_samples = 5000
    data_x = data_x[:n_samples]
    data_y = data_y[:n_samples]

    ts_data_x = Tensor(data_x)
    ts_data_y = Tensor(data_y)

    out_c = data_y.shape[-1]

    lr = 1e-2
    batch_size = 24
    epochs = 1000
    epochs_show = 5

    ts_kernels1 = Tensor(n.random.randn(4, 1, 3, 3) / 5, is_grad=True)
    ts_bias1 = Tensor(n.zeros((4,)), is_grad=True)
    ts_kernels2 = Tensor(n.random.randn(4, 4, 3, 3) / 8, is_grad=True)
    ts_bias2 = Tensor(n.zeros((4,)), is_grad=True)
    ts_weights3 = Tensor(n.random.randn(400, out_c) / n.sqrt(400 + out_c), is_grad=True)
    ts_bias3 = Tensor(n.zeros((out_c, )), is_grad=True)
    c = Tensor(n.array([1 / batch_size, ]))

    batch_i = n.random.randint(0, n_samples, batch_size)
    ts_batch_x = ts_data_x[batch_i]
    ts_batch_y = ts_data_y[batch_i]
    y1 = Sigmoid(Conv2d(ts_batch_x, ts_kernels1, ts_bias1)())()
    y2 = Sigmoid(Conv2d(y1, ts_kernels2, ts_bias2)())()
    y3 = Flatten(y2)()
    y4 = y3 @ ts_weights3 + ts_bias3
    y5 = Sigmoid(y4)()
    loss = c * (ts_batch_y - y5) * (ts_batch_y - y5)

    g = n.ones(loss.shape)

    for i in range(epochs):

        batch_i = n.random.randint(0, n_samples, batch_size)
        # ts_batch_x = Tensor(data_x[batch_i])
        # ts_batch_y = Tensor(data_y[batch_i])
        ts_batch_x = ts_data_x[batch_i]
        ts_batch_y = ts_data_y[batch_i]

        y1 = Sigmoid(Conv2d(ts_batch_x, ts_kernels1, ts_bias1)())()
        y2 = Sigmoid(Conv2d(y1, ts_kernels2, ts_bias2)())()
        y3 = Flatten(y2)()
        y4 = y3 @ ts_weights3 + ts_bias3
        y5 = Sigmoid(y4)()
        loss = c * (ts_batch_y - y5) * (ts_batch_y - y5)

        loss.backward(g)

        ts_kernels1.arr += adam(ts_kernels1.arr, ts_kernels1.grad, lr, i+1)
        ts_kernels2.arr += adam(ts_kernels2.arr, ts_kernels2.grad, lr, i+1)
        ts_weights3.arr += adam(ts_weights3.arr, ts_weights3.grad, lr, i+1)
        ts_bias1.arr += adam(ts_bias1.arr, ts_bias1.grad, lr, i+1)
        ts_bias2.arr += adam(ts_bias2.arr, ts_bias2.grad, lr, i+1)
        ts_bias3.arr += adam(ts_bias3.arr, ts_bias3.grad, lr, i+1)

        ts_kernels1.grad_zeros()
        ts_bias1.grad_zeros()
        ts_kernels2.grad_zeros()
        ts_bias2.grad_zeros()
        ts_weights3.grad_zeros()
        ts_bias3.grad_zeros()

        if (i+1) % epochs_show == 0:
            print('{} loss - {}'.format(i + 1, n.sum(loss.arr)))

    batch_i = n.random.randint(0, n_samples, batch_size)
    ts_batch_x = ts_data_x[batch_i]
    ts_batch_y = ts_data_y[batch_i]
    y1 = Sigmoid(Conv2d(ts_batch_x, ts_kernels1, ts_bias1)())()
    y2 = Sigmoid(Conv2d(y1, ts_kernels2, ts_bias2)())()
    y3 = Flatten(y2)()
    y4 = y3 @ ts_weights3 + ts_bias3
    y5 = Sigmoid(y4)()

    print()
    ls1 = [n.argmax(i) for i in ts_batch_y.arr]
    ls2 = [n.argmax(i) for i in y5.arr]
    print(ls1)
    print(ls2)


if __name__ == '__main__':
    test2()




