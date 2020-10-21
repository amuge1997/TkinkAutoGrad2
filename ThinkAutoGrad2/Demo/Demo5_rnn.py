import numpy as n
n.random.seed(0)

from ThinkAutoGrad2.Core import backward, Optimizer, Losses, Tensor, Layers, Init


if __name__ == '__main__':

    hidden_size = 8
    u = Init.xavier((1, hidden_size), is_grad=True)
    w = Init.xavier((hidden_size, hidden_size), is_grad=True)
    v = Init.xavier((hidden_size, 1), is_grad=True)
    b = Init.zeros((hidden_size,), is_grad=True)

    vb = Init.zeros((1,), is_grad=True)

    t = n.linspace(0, 10, 200)
    sin_x = n.sin(t)
    max_time_step = sin_x.shape[0]

    batch = 30
    time_step = 10
    epoch = 1500
    mse = Losses.MSE()
    adam = Optimizer.Adam(2e-4)
    h = Init.zeros((batch, 1, hidden_size))
    loss_record = []
    for i in range(epoch):
        x_ls = []   # 输入
        y_ls = []   # 真实输出
        for bah in range(batch):
            start = n.random.randint(0, max_time_step-time_step - 1)
            end = start + time_step
            bah_x = sin_x[start:end]
            x_ls.append(bah_x)
            bah_y = sin_x[end + 1, n.newaxis]
            y_ls.append(bah_y)
        x = n.concatenate(x_ls)
        x = x.reshape((batch, time_step, 1))
        x = Tensor(x)
        y = n.concatenate(y_ls)
        y = y.reshape((batch, 1, 1))
        y = Tensor(y)

        oh = Layers.RNN(x, h, u, w, b)()
        yp = oh @ v + vb
        loss = mse(yp, y)

        for z in [u, w, v, vb]:
            z.grad_zeros()
        backward(loss)
        adam.run([u, w, v, vb])

        print('epoch {} - {}'.format(i+1, n.mean(loss.arr)))
        loss_record.append(n.mean(loss.arr))

    h = Init.zeros((1, 1, hidden_size))
    yp_n = max_time_step - time_step
    yp_arr = n.zeros((yp_n, 1))
    for i in range(0, yp_n):
        x = sin_x[i:i + time_step].reshape((1, time_step, 1))
        x = Tensor(x)
        oh = Layers.RNN(x, h, u, w, b)()
        y = oh @ v + vb
        yp_arr[i] = y.arr[0, 0]
    yr = sin_x[0 + time_step + 1:time_step + yp_n + 1]

    import matplotlib.pyplot as p
    p.grid()
    p.plot(yp_arr, c='blue', label='pred')
    p.plot(yr, c='red', label='real')
    p.legend(prop={'size': 16, 'weight': 'bold'})

    p.figure()
    p.grid()
    p.title('train loss', fontdict={'weight': 'bold', 'size': 16})
    p.plot(loss_record, c='blue')

    p.show()












