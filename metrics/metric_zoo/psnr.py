import paddle
import paddle.nn as nn


class PSNR(nn.Layer):
    def __init__(self, data_range=1., **kwargs):
        super(PSNR, self).__init__()
        self.data_range = data_range

    def forward(self, pred, real):
        assert (real.min().item() >= 0) & (real.max().item() <= self.data_range)
        pred = paddle.clip(255./self.data_range*pred, 0, 255).astype('float32')
        real = paddle.clip(255./self.data_range*real, 0, 255).astype('float32')
        diff = pred - real
        mse = paddle.square(diff).mean()
        psnr = 10*paddle.log10(255*255/mse)
        
        return {"PSNR":psnr}