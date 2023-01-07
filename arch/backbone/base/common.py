import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2D(in_channels=in_channels, 
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding=(kernel_size//2),
                     bias_attr=bias)

class MeanShift(nn.Conv2D):
    def __init__(self,
                 rgb_range,
                 rgb_mean=[0.4488, 0.4371, 0.4040],
                 rgb_std=[1.0, 1.0, 1.0],
                 sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = paddle.to_tensor(rgb_std)
        self.weight.set_value(paddle.eye(3).reshape((3,3,1,1)) / std.reshape((3,1,1,1)))
        self.bias.set_value(sign * rgb_range * paddle.to_tensor(rgb_mean) / std)
        for p in self.parameters():
            p.stop_gradient = True


class BasicBlock(nn.Sequential):
    def __init__(self,
                 conv,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 bias=False,
                 bn=True,
                 act=nn.ReLU()):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2D(out_channels))
        if act is not None:
            m.append(act)
        
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Layer):
    def __init__(self,
                 conv,
                 n_feats,
                 kernel_size,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(),
                 res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2D(n_feats, n_feats, kernel_size=kernel_size, bias_attr=True, padding=1))
            # m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2D(n_feats))
            if i == 0:
                m.append(act)
        
        # self.body = nn.Sequential(*m)
        self.body = nn.LayerList(m)
        self.res_scale = res_scale

    def forward(self, x):
        from reprod_log import ReprodLogger
        log = ReprodLogger()
        log.add('x', x.detach().cpu().numpy())
        res = x
        print(res)
        for i, layer in enumerate(self.body):
            res = layer(res)
            print(res)
            assert 1==0
            log.add(f'res_{i}', res.numpy())
        res *= self.res_scale
        # res = self.body(x) * self.res_scale
        log.add('res', res.detach().cpu().numpy())
        res += x
        log.add('res+x', res.detach().cpu().numpy())
        log.save('/mnt/zh/align/edsr/paddle/ResBlock')
        assert 1==0
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2D(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU())
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2D(n_feats))
            if act == 'relu':
                m.append(nn.ReLU())
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)