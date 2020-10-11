import numpy as n
n.random.seed(0)

from ThinkAutoGrad2.Model import Model
from ThinkAutoGrad2.Backward import backward
from ThinkAutoGrad2.Losses import MSE
from ThinkAutoGrad2.Conv2d import Conv2d
from ThinkAutoGrad2.Utils import Flatten
from ThinkAutoGrad2.Tensor import Tensor
from ThinkAutoGrad2.Activate import Sigmoid, Relu
from ThinkAutoGrad2.Optimizer import Adam
from sklearn.metrics import accuracy_score


class Net(Model):
    def __init__(self, out_c):
        super(Net, self).__init__()
        ts_kernels1 = Tensor(n.random.randn(4, 1, 2, 2) / n.sqrt(4 + 1), is_grad=True)
        ts_bias1 = Tensor(n.zeros((4,)), is_grad=True)

        ts_kernels2 = Tensor(n.random.randn(8, 4, 2, 2) / n.sqrt(8 + 4), is_grad=True)
        ts_bias2 = Tensor(n.zeros((8,)), is_grad=True)

        ts_kernels4 = Tensor(n.random.randn(16, 8, 2, 2) / n.sqrt(16 + 8), is_grad=True)
        ts_bias4 = Tensor(n.zeros((16,)), is_grad=True)

        ts_kernels5 = Tensor(n.random.randn(32, 16, 3, 3) / n.sqrt(32 + 16), is_grad=True)
        ts_bias5 = Tensor(n.zeros((32,)), is_grad=True)

        ts_weights3 = Tensor(n.random.randn(32 * 4 * 4, out_c) / n.sqrt(32 * 4 * 4 + out_c), is_grad=True)
        ts_bias3 = Tensor(n.zeros((out_c,)), is_grad=True)

        self.weights_list = [
            ts_kernels1, ts_bias1, ts_kernels2, ts_bias2,
            ts_kernels4, ts_bias4, ts_kernels5, ts_bias5,
            ts_weights3, ts_bias3
        ]

    def forward(self, inps):
        ts_kernels1, ts_bias1, ts_kernels2, ts_bias2, ts_kernels4, ts_bias4, ts_kernels5, ts_bias5, ts_weights3, ts_bias3 = self.weights_list
        y1 = Relu(Conv2d(inps, ts_kernels1, ts_bias1, stride=(2, 2), is_padding=False)())()  # 16
        y2 = Relu(Conv2d(y1, ts_kernels2, ts_bias2, stride=(2, 2), is_padding=False)())()  # 8
        y6 = Relu(Conv2d(y2, ts_kernels4, ts_bias4, stride=(2, 2), is_padding=False)())()  # 4
        y7 = Relu(Conv2d(y6, ts_kernels5, ts_bias5, stride=(1, 1))())()  # 4
        y3 = Flatten(y7)()
        y4 = y3 @ ts_weights3 + ts_bias3
        y5 = Sigmoid(y4)()
        return y5


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

    data_x = data_x
    data_y = data_y

    cls = n.max(data_y) + 1
    data_y = one_hot_encoding(data_y, cls)

    return data_x, data_y


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
    epochs = 500
    epochs_show = 10

    net = Net(out_c)
    mse = MSE()
    adam = Adam(lr)

    for i in range(epochs):

        batch_i = n.random.randint(0, n_samples, batch_size)
        ts_batch_x = ts_data_x[batch_i]
        ts_batch_y = ts_data_y[batch_i]

        predict_y = net.forward(ts_batch_x)
        loss = mse(predict_y, ts_batch_y)
        backward(loss)

        adam.run(net.get_weights())
        net.grad_zeros()

        if (i+1) % epochs_show == 0:
            print('{} loss - {}'.format(i + 1, n.sum(loss.arr)))

    batch_i = n.array(range(32))
    ts_batch_x = ts_data_x[batch_i]
    ts_batch_y = ts_data_y[batch_i]
    predict_y = net.forward(ts_batch_x)

    print()
    ls1 = [n.argmax(i) for i in ts_batch_y.arr]
    ls2 = [n.argmax(i) for i in predict_y.arr]
    print(ls1)
    print(ls2)
    print()
    acc = accuracy_score(ls1, ls2)
    print('acc - {}'.format(n.round(acc, 3)))


if __name__ == '__main__':
    test2()




