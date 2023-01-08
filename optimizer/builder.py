from paddle.optimizer import SGD, Adam, Momentum
from ._lr import *
import copy


def build_optimizer(engine, cfg):
    cfg = copy.deepcopy(cfg['Optimizer'])
    name = cfg.pop('name')
    _lr = cfg.pop('learning_rate')
    _lr_name = _lr.pop('name')
    lr = eval(_lr_name)(**_lr)
    opt = eval(name)(parameters=engine.model.parameters(),
                     learning_rate=lr)
    
    return opt, lr