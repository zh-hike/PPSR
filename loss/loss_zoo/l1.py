import paddle.nn as nn


class L1Loss(nn.Layer):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, inputs, label):
        return self.cri(inputs, label)