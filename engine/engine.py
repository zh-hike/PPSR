import paddle
from arch.builder import build_arch
import paddle.distributed as dist
import paddle
import os
from utils.logger import Logger
from utils.average_meter import AverageMeter, AverageMeterDict
from .evaluation.util import log_eval_info
from .trainer import *
from .evaluation import *
from tqdm import tqdm
from dataloader.builder import build_dataloader
from loss.builder import build_loss
from optimizer.builder import build_optimizer
from .trainer.util import save_checkoutpoints
from metrics.builder import build_metric


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self._name = f"{cfg['Arch']['name']}_{cfg['Data']['Train']['Dataset']['name']}"
        # 准备
        os.makedirs(os.path.join(self.cfg['Global']['output_dir'], self.cfg['Arch']['name']), exist_ok=True)

        #  set model
        self.model = build_arch(cfg['Arch'])
        if self.cfg['Global'].get('pretrained_model', None):
            self.model.set_dict(paddle.load(self.cfg['Global']['pretrained_model']))
        if self.cfg['Global'].get('dist', False):
            dist.init_parallel_env()
            self.model = paddle.DataParallel(self.model)
        
        # logger
        output_dir = cfg['Global'].get('output_dir', 'output_dir')
        self.train_logger = Logger(logger_file=f"./{output_dir}/{cfg['Arch']['name']}/train.log")
        self.train_logger.print_config(cfg)
        self.eval_logger = Logger(logger_file=f"./{output_dir}/{cfg['Arch']['name']}/eval.log")
        
        # time info
        self.time_info = {'read_cost': AverageMeter(name="read_cost", postfix="s"),
                          'batch_cost': AverageMeter(name="batch_cost", postfix="s")}

        # train func
        self.train_func = eval("train_epoch_" + cfg['Global']['trainer'])
        self.eval_func = eval("eval_epoch_" + cfg['Global'].get("evaler", "base"))

        # dataloader
        self.train_dl = build_dataloader(cfg, mode='train')
        self.eval_dl = build_dataloader(cfg, mode='eval')

        # loss_func
        self.train_loss_func = build_loss(self.cfg)
        self.train_loss_info = AverageMeterDict(names=[list(d)[0] for d in self.cfg['Loss']['Train']]+['loss'])
        if self.cfg['Loss'].get('Eval', False):
            self.eval_loss_func = build_loss(self.cfg, mode='eval')
            self.eval_loss_info = AverageMeterDict(names=[list(d)[0] for d in self.cfg['Loss']['Train']]+['loss'])

        # optimizer
        self.opt, self.lr = build_optimizer(self, cfg)
        self.schedule_update_by = cfg['Global'].get('schedule_update_by', 'step')
        assert self.schedule_update_by in ['step', 'epoch']

        # metric
        self.best_metric_value = 0
        if self.cfg['Metric'].get('Train', False):
            self.train_metric_func = build_metric(self.cfg)
            self.train_metric_info = AverageMeterDict(names=[list(d)[0] for d in self.cfg['Metric']['Train']])
        self.eval_metric_func = build_metric(self.cfg, mode="eval")
        self.eval_metric_info = AverageMeterDict(names=[list(d)[0] for d in self.cfg['Metric']['Eval']])


    def train(self):
        """
        模型训练
        """
        bar_disable = self.cfg['Global'].get('bar_disable', True)
        for epoch_id in tqdm(range(self.cfg['Global']['epochs']), ncols=90, disable=bar_disable):
            
            self.train_func(self, epoch_id)
            self.save_checkpoints(epoch_id, 0, True)
            self.eval()
            if self.best_metric_value < self.eval_metric_info.amd[self.cfg['Metric']['save_rely_metric']].avg:
                self.best_metric_value = self.eval_metric_info.amd[self.cfg['Metric']['save_rely_metric']].avg
                self.save_model('best')
            
            self.save_model()


    def eval(self):
        """
        模型评估
        """
        self.model.eval()
        self.eval_func(self)
        self.model.train()
        log_eval_info(self)
        
    def info_reset(self):
        self.time_info['read_cost'].reset()
        self.time_info['batch_cost'].reset()
        self.train_loss_info.reset()
        if getattr(self, "eval_loss_info", False):
            self.eval_loss_info.reset()
        if getattr(self, "train_metric_info", False):
            self.train_metric_info.reset()
        self.eval_metric_info.reset()

    def save_model(self, mode='latest'):
        """
        保存模型参数，包括bese model和latest model
        """
        assert mode in ['latest', 'best']
        if mode == 'latest':
            model_name = 'latest_model.pdparams'
            opt_name = 'latest_opt.pdopt'
        elif mode == 'best':
            model_name = 'best_model.pdparams'
            opt_name = 'best_opt.pdopt'
        paddle.save(self.model.state_dict(), os.path.join(self.cfg['Global']['output_dir'], self.cfg['Arch']['name'], model_name))
        paddle.save(self.opt.state_dict(), os.path.join(self.cfg['Global']['output_dir'], self.cfg['Arch']['name'], opt_name))


    def save_checkpoints(self, epoch_id, iter_id, force=False):
        if force:
            save_checkoutpoints(self, epoch_id, iter_id)
        else:
            if iter_id % self.cfg['Global']['save_interval_step'] == 0:
                save_checkoutpoints(self, epoch_id, iter_id)

