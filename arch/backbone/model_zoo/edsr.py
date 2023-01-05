from ..base import common
from ..base.theseus_layer import TheseusLayer
import paddle.nn as nn


class EDSRModel(TheseusLayer):
    def __init__(self, 
                 n_resblocks,
                 n_feats,
                 n_colors,
                 res_scale,
                 scale,
                 rgb_range,
                 conv=common.default_conv):

        super(EDSRModel, self).__init__()
        kernel_size = 3
        scale0 = scale[0]
        act = nn.ReLU()
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in (n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x


def EDSR(n_resblocks,
         n_feats,
         n_colors,
         res_scale,
         scale,
         rgb_range,
         **kwargs):
    return EDSRModel(n_resblocks,
                     n_feats,
                     n_colors,
                     res_scale,
                     scale,
                     rgb_range)
                     