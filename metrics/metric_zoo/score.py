from .ms_ssim import MSSSIM
from .psnr import PSNR
import paddle.nn as nn
import copy

class Score(nn.Layer):
    def __init__(self, **kwargs):
        super(Score, self).__init__()
        cfg = copy.deepcopy(kwargs)
        self.metric_funcs = []
        self.weights = []
        for k, v in cfg.items():
            weight = v.pop('weight', 1)
            self.weights.append(weight)
            self.metric_funcs.append(eval(k)(**v))

    def forward(self, pred, real):
        score = 0
        results = {}
        for weight, func in zip(self.weights, self.metric_funcs):
            metric = func(pred, real)
            score += weight * metric
            results[func.__class__.__name__] = metric

        results['Score'] = score
        return results 
        

