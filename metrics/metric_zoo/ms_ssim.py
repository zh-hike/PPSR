from paddle_msssim import ms_ssim
import paddle.nn as nn
import paddle


class MSSSIM(nn.Layer):
    def __init__(self, 
                 data_range=1,
                 win_size=11,
                 **kwargs):
        super(MSSSIM, self).__init__()
        self.data_range = data_range
        self.win_size = win_size
        self.cfg = kwargs

    def forward(self, pred, real):
        pred = paddle.clip(pred, -self.data_range, self.data_range)
        assert (real.min().item() >= (-self.data_range)) & (real.max().item() <= self.data_range)
        return {"MSSSIM":ms_ssim(pred, real, data_range=self.data_range, win_size=self.win_size)}