import paddle
from arch.builder import build_arch
import paddle.distributed as dist
import paddle
from utils import logger
from utils.average_meter import AverageMeter
from .trainer import *
from tqdm import tqdm
from dataloader.builder import build_dataloader
from loss.builder import build_loss
from optimizer.builder import build_optimizer
from .trainer.util import save_checkoutpoints


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self._name = f"{cfg['Arch']['name']}_{cfg['Data']['Train']['Dataset']['name']}"
        #  set model
        self.model = build_arch(cfg['Arch'])
        if self.cfg['Global'].get('dist', False):
            dist.init_parallel_env()
            self.model = paddle.DataParallel(self.model)
        
        # logger
        logger.init_logger(logger_file=f"./output/{cfg['Arch']['name']}/train.log")
        logger.print_config(cfg)

        # time info
        self.time_info = {'read_cost': AverageMeter(name="read_cost", postfix="s"),
                          'batch_cost': AverageMeter(name="batch_cost", postfix="s")}

        # train func
        self.train_func = eval("train_epoch_" + cfg['Global']['trainer'])

        # dataloader
        self.train_dl = build_dataloader(cfg, mode='train')

        # loss_func
        self.train_loss_func = build_loss(self.cfg)
        self.train_loss_info = AverageMeter(name="loss", postfix="")


        # optimizer
        self.opt, self.lr = build_optimizer(self, cfg)
        self.schedule_update_by = cfg['Global'].get('schedule_update_by', 'step')
        assert self.schedule_update_by in ['step', 'epoch']

    def train(self):
        bar_disable = self.cfg['Global'].get('bar_disable', True)
        for epoch_id in tqdm(range(self.cfg['Global']['epochs']), ncols=90, disable=bar_disable):
            
            self.train_func(self, epoch_id)
            self.save_checkpoints(epoch_id, 0, True)

    def save_checkpoints(self, epoch_id, iter_id, force=False):
        if force:
            save_checkoutpoints(self, epoch_id, iter_id)
        else:
            if iter_id % self.cfg['Global']['save_interval_step'] == 0:
                save_checkoutpoints(self, epoch_id, iter_id)

