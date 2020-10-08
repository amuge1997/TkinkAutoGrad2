from ThinkAutoGrad2.Utils_conv import Conv2d
from ThinkAutoGrad2.Utils import Flatten
from ThinkAutoGrad2.Tensor import Tensor
from ThinkAutoGrad2.Activate import Sigmoid, Relu
import numpy as n
from sklearn.metrics import accuracy_score


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
        x = cv.resize(i, (32, 32))
        x = x[n.newaxis, ...]
        x_ls.append(x)
    data_x = n.concatenate(x_ls)

    data_x = data_x#[(data_y==0)|(data_y==1)|(data_y==2)|(data_y==3)|(data_y==4)|(data_y==5)|(data_y==6)]
    data_y = data_y#[(data_y==0)|(data_y==1)|(data_y==2)|(data_y==3)|(data_y==4)|(data_y==5)|(data_y==6)]

    cls = n.max(data_y) + 1
    data_y = one_hot_encoding(data_y, cls)

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

    n_samples = 1000
    data_x = data_x[:n_samples]
    data_y = data_y[:n_samples]

    ts_data_x = Tensor(data_x)
    ts_data_y = Tensor(data_y)

    out_c = data_y.shape[-1]

    lr = 1e-3
    batch_size = 24
    epochs = 1000
    epochs_show = 10

    ts_kernels1 = Tensor(n.random.randn(4, 1, 2, 2) / n.sqrt(4+1), is_grad=True)
    ts_bias1 = Tensor(n.zeros((4,)), is_grad=True)
    ts_kernels2 = Tensor(n.random.randn(8, 4, 2, 2) / n.sqrt(8+4), is_grad=True)
    ts_bias2 = Tensor(n.zeros((8,)), is_grad=True)

    ts_kernels4 = Tensor(n.random.randn(16, 8, 2, 2) / n.sqrt(16+8), is_grad=True)
    ts_bias4 = Tensor(n.zeros((16,)), is_grad=True)

    ts_kernels5 = Tensor(n.random.randn(32, 16, 3, 3) / n.sqrt(32+16), is_grad=True)
    ts_bias5 = Tensor(n.zeros((32,)), is_grad=True)

    ts_weights3 = Tensor(n.random.randn(32*4*4, out_c) / n.sqrt(32*4*4 + out_c), is_grad=True)
    ts_bias3 = Tensor(n.zeros((out_c, )), is_grad=True)
    c = Tensor(n.array([1 / batch_size, ]))

    batch_i = n.random.randint(0, n_samples, batch_size)
    ts_batch_x = ts_data_x[batch_i]
    ts_batch_y = ts_data_y[batch_i]
    y1 = Relu(Conv2d(ts_batch_x, ts_kernels1, ts_bias1, stride=(2, 2), is_padding=False)())()      # 16
    y2 = Relu(Conv2d(y1, ts_kernels2, ts_bias2, stride=(2, 2), is_padding=False)())()              # 8
    y6 = Relu(Conv2d(y2, ts_kernels4, ts_bias4, stride=(2, 2), is_padding=False)())()              # 4
    y7 = Relu(Conv2d(y6, ts_kernels5, ts_bias5, stride=(1, 1))())()                                 # 4
    y3 = Flatten(y7)()
    y4 = y3 @ ts_weights3 + ts_bias3
    y5 = Sigmoid(y4)()
    loss = c * (ts_batch_y - y5) * (ts_batch_y - y5)

    g = n.ones(loss.shape)

    for i in range(epochs):

        batch_i = n.random.randint(0, n_samples, batch_size)
        ts_batch_x = ts_data_x[batch_i]
        ts_batch_y = ts_data_y[batch_i]

        y1 = Relu(Conv2d(ts_batch_x, ts_kernels1, ts_bias1, stride=(2, 2), is_padding=False)())()  # 16
        y2 = Relu(Conv2d(y1, ts_kernels2, ts_bias2, stride=(2, 2), is_padding=False)())()  # 8
        y6 = Relu(Conv2d(y2, ts_kernels4, ts_bias4, stride=(2, 2), is_padding=False)())()  # 4
        y7 = Relu(Conv2d(y6, ts_kernels5, ts_bias5, stride=(1, 1))())()  # 4
        y3 = Flatten(y7)()
        y4 = y3 @ ts_weights3 + ts_bias3
        y5 = Sigmoid(y4)()
        loss = c * (ts_batch_y - y5) * (ts_batch_y - y5)

        loss.backward(g)

        ts_kernels1.arr += adam(ts_kernels1.arr, ts_kernels1.grad, lr, i+1)
        ts_kernels2.arr += adam(ts_kernels2.arr, ts_kernels2.grad, lr, i+1)
        ts_kernels4.arr += adam(ts_kernels4.arr, ts_kernels4.grad, lr, i + 1)
        ts_kernels5.arr += adam(ts_kernels5.arr, ts_kernels5.grad, lr, i + 1)
        ts_bias1.arr += adam(ts_bias1.arr, ts_bias1.grad, lr, i+1)
        ts_bias2.arr += adam(ts_bias2.arr, ts_bias2.grad, lr, i+1)
        ts_bias4.arr += adam(ts_bias4.arr, ts_bias4.grad, lr, i + 1)
        ts_bias5.arr += adam(ts_bias5.arr, ts_bias5.grad, lr, i + 1)

        ts_weights3.arr += adam(ts_weights3.arr, ts_weights3.grad, lr, i + 1)
        ts_bias3.arr += adam(ts_bias3.arr, ts_bias3.grad, lr, i + 1)

        ts_kernels1.grad_zeros()
        ts_bias1.grad_zeros()
        ts_kernels2.grad_zeros()
        ts_bias2.grad_zeros()
        ts_weights3.grad_zeros()
        ts_bias3.grad_zeros()
        ts_kernels4.grad_zeros()
        ts_bias4.grad_zeros()
        ts_kernels5.grad_zeros()
        ts_bias5.grad_zeros()

        if (i+1) % epochs_show == 0:
            print('{} loss - {}'.format(i + 1, n.sum(loss.arr)))

    batch_i = n.array(range(32))
    ts_batch_x = ts_data_x[batch_i]
    ts_batch_y = ts_data_y[batch_i]
    y1 = Relu(Conv2d(ts_batch_x, ts_kernels1, ts_bias1, stride=(2, 2), is_padding=False)())()  # 16
    y2 = Relu(Conv2d(y1, ts_kernels2, ts_bias2, stride=(2, 2), is_padding=False)())()  # 8
    y6 = Relu(Conv2d(y2, ts_kernels4, ts_bias4, stride=(2, 2), is_padding=False)())()  # 4
    y7 = Relu(Conv2d(y6, ts_kernels5, ts_bias5, stride=(1, 1))())()  # 4
    y3 = Flatten(y7)()
    y4 = y3 @ ts_weights3 + ts_bias3
    y5 = Sigmoid(y4)()

    print()
    ls1 = [n.argmax(i) for i in ts_batch_y.arr]
    ls2 = [n.argmax(i) for i in y5.arr]
    print(ls1)
    print(ls2)
    print()
    acc = accuracy_score(ls1, ls2)
    print('acc - {}'.format(n.round(acc, 3)))

    # 全连接16个单元,训练卷积层   acc = 0.781
    # 全连接16个单元,不训练卷积层  acc = 0.156


if __name__ == '__main__':
    test2()




