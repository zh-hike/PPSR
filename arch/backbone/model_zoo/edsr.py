from ..base import common
from ..base.theseus_layer import TheseusLayer
import paddle.nn as nn
import collections.abc
import logging

class EDSRModel(TheseusLayer):
    def __init__(self, 
                 n_resblocks,
                 n_feats,
                 n_colors,
                 res_scale,
                 scale,
                 rgb_range,
                 conv=common.default_conv,
                 num_fpn_layers: int=4):

        super(EDSRModel, self).__init__()
        self.num_fpn_layers = num_fpn_layers
        kernel_size = 3
        if not isinstance(scale, collections.abc.Iterable):
            scale = (scale,)
        scale = scale[0]
        act = nn.ReLU()
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        self.m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        self.m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]
        self.feats = []
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*self.m_body)
        self.tail = nn.Sequential(*m_tail)
        self._hook_layer()

    def _hook(self, layer, inputs, output):
        self.feats.append(output)

    def _hook_layer(self):
        for layer in self.m_body:
            layer.register_forward_post_hook(self._hook)

    def load_model_from_torch(self, model_path="/mnt/zh/align/edsr/torch_model.pt"):
        keys = []
        for key in self.state_dict():
            keys.append(key)
        pd_state_dict = {}
        import torch
        import paddle
        from reprod_log import ReprodLogger
        log = ReprodLogger()
        torch_state_dict = torch.load(model_path)
        for key in keys:
            print(key)
            torch_key = key.replace('EDSRModel', 'EDSR')
            if 'bn' in torch_key and '_mean' in torch_key:
                torch_key = torch_key.replace('_mean', 'running_mean')
            if 'bn' in torch_key and '_variance' in torch_key:
                torch_key = torch_key.replace('_variance', 'running_var')
            if not torch_key in torch_state_dict:
                print(f"{torch_key} load fail")
            pd_state_dict[key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())
            log.add(torch_key, pd_state_dict[key].numpy())
        
        self.set_state_dict(pd_state_dict)
        print("保存模型参数。。。")
        log.save('/mnt/zh/align/edsr/paddle/paddle_state')
    
    def _clear_hook(self):
        self.feats.clear()

    def forward(self, x):
        x = self.sub_mean(x)
        
        x = self.head(x)
        res = self.body(x)
        res += x
        for feat in self.feats[:-(self.num_fpn_layers+1):-1]:
            res += feat
        x = self.tail(res)
        x = self.add_mean(x)
        self._clear_hook()
        return x


def EDSR(n_resblocks=16,
         n_feats=64,
         n_colors=3,
         res_scale=1,
         scale=2,
         rgb_range=255.0,
         **kwargs) -> nn.Layer:
    return EDSRModel(n_resblocks,
                     n_feats,
                     n_colors,
                     res_scale,
                     scale,
                     rgb_range)
                     
