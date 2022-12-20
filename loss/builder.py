from .loss_zoo import *
import copy
import paddle.nn as nn


class CombineLoss(nn.Layer):
    def __init__(self, loss_list):
        super(CombineLoss, self).__init__()
        self.loss_funs = []
        self.loss_weights = []
        self.loss_name = []
        for loss_cfg in loss_list:
            name = list(loss_cfg)[0]
            cfg = loss_cfg[name]
            self.loss_name.append(name)
            weight = cfg.pop('weight')
            self.loss_weights.append(weight)
            self.loss_funs.append(eval(name)(**cfg))

    def forward(self, pred, targets):
        loss_dict = {}
        losses = 0
        for name, weight, loss_func in zip(self.loss_name, self.loss_weights, self.loss_funs):
            loss = weight * loss_func(pred, targets).mean()
            losses += loss
            loss_dict[name] = loss
        loss_dict['loss'] = losses
        return loss_dict

def build_loss(config, mode='train'):
    assert mode in ['train', 'eval']
    cfg = copy.deepcopy(config)
    if mode == 'train':
        cfg = cfg['Loss']['Train']
    else:
        cfg = cfg['Loss']['Eval']

    return CombineLoss(cfg)
    
    