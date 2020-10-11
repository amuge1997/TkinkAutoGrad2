

class Model:
    def __init__(self):
        self.weights_list = []

    def grad_zeros(self):
        for w in self.weights_list:
            w.grad_zeros()

    def get_weights(self):
        return self.weights_list

    def forward(self, inps):
        pass





