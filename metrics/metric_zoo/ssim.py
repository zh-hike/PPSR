from paddle_msssim import ssim
import paddle.nn as nn
import paddle


class SSIM(nn.Layer):
    def __init__(self, 
                 data_range=1,
                 win_size=11,
                 **kwargs):
        super(SSIM, self).__init__()
        self.data_range = data_range
        self.win_size = win_size
        self.cfg = kwargs

    def forward(self, pred, real):
        pred = paddle.clip(pred, -self.data_range, self.data_range)
        assert (real.min().item() >= (-self.data_range)) & (real.max().item() <= self.data_range)
        return {"SSIM":ssim(pred, real, data_range=self.data_range, win_size=self.win_size)}