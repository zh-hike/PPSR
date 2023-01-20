import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import collections.abc

def _calculate_fan_in_fan_out(shape):
    assert isinstance(shape, collections.abc.Iterable), "shape must be a iterable"
    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    for s in shape[2:]:
        receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    fan_in, _ = _calculate_fan_in_fan_out((out_channels, in_channels, kernel_size, kernel_size))
    bound = 1 / math.sqrt(fan_in)
    bias_init = nn.initializer.Uniform(-bound, bound) if bias else bias
    
    return nn.Conv2D(in_channels=in_channels, 
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding=(kernel_size//2),
                     bias_attr=bias_init,
                     weight_attr=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity='leaky_relu')
                     )

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
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2D(n_feats))
            if i == 0:
                m.append(act)
        
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res = res + x
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