from .data import *
import copy
from paddle.io import DistributedBatchSampler, BatchSampler
from paddle.io import DataLoader


def build_dataloader(cfg, mode='train'):
    assert mode in ['train', 'eval'], "mode 必须是train 或者 eval"
    if mode == 'train':
        cfg = copy.deepcopy(cfg['Data']['Train'])
    else:
        cfg = copy.deepcopy(cfg['Data']['Eval'])

    dataset_cfg = cfg['Dataset']
    name = dataset_cfg.pop('name')
    dataset = eval(name)(**dataset_cfg)

    dataloader_cfg = cfg['DataLoader']
    sampler_cfg = dataloader_cfg.pop('batch_sampler')
    sampler_name = sampler_cfg.pop('name')
    batch_sampler = eval(sampler_name)(dataset=dataset, **sampler_cfg)

    dl = DataLoader(dataset=dataset, batch_sampler=batch_sampler, **dataloader_cfg)

    return dl