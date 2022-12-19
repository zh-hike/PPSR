from .backbone import *
import copy
import paddle.nn as nn


def build_arch(config):
    cfg = copy.deepcopy(config)
    name = cfg.pop('name')
    use_sync_bn = cfg.pop('use_sync_bn', False)
    model = eval(name)(**cfg)
    if use_sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model