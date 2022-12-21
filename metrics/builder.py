from .metric_zoo import *
import paddle.nn as nn
import copy


class CombineMetric(nn.Layer):
    def __init__(self, metric_list):
        super(CombineMetric, self).__init__()
        self.metric_func = []
        for metric_cfg in metric_list:
            cfg = copy.deepcopy(metric_cfg)
            name = list(cfg)[0]
            self.metric_func.append(eval(name)(**cfg[name]))

    def forward(self, pred, real):
        results = {}
        for func in self.metric_func:
            results.update(func(pred, real))

        return results


def build_metric(config, mode='train'):
    assert mode in ['train', 'eval']
    cfg = copy.deepcopy(config)
    if mode == 'train':
        if cfg['Metric'].get('Train', None) is None:
            return None
        return CombineMetric(cfg['Metric']['Train'])
    else:
        assert cfg['Metric'].get('Eval', False) != False, "评估必须设置验证参数"
        return CombineMetric(cfg['Metric']['Eval'])