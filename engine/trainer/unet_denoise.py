import time
from .util import log_train_info, save_checkoutpoints
import paddle


def train_epoch_unet_denoise(engine, epoch_id, iter_start=0):
    start_time = time.time()
    step_per_epoch = engine.cfg['Global']['step_per_epoch']
    train_dataloader_iter = iter(engine.train_dl)
    for iter_id in range(iter_start, step_per_epoch):
        try:
            batch = next(train_dataloader_iter)
        except:
            train_dataloader_iter = iter(engine.train_dl)
            batch = next(train_dataloader_iter)

        inputs, targets = batch
        engine.time_info['read_cost'].update(time.time() - start_time)
        pred = engine.model(inputs)
        
        loss_dict = engine.train_loss_func(pred, targets)
        loss = loss_dict['loss']
        
        engine.opt.clear_grad()
        loss.backward()
        engine.opt.step()
        engine.train_loss_info.update(loss.item())
        if engine.schedule_update_by == 'step':
            engine.lr.step()

        if iter_id % engine.cfg['Global']['print_batch_step'] == 0:
            log_train_info(engine, epoch_id, iter_id)
            engine.time_info['read_cost'].reset()
            engine.time_info['batch_cost'].reset()
            engine.train_loss_info.reset()

        engine.time_info['batch_cost'].update(time.time() - start_time)
        start_time = time.time()
        engine.save_checkpoints(epoch_id, iter_id)


    if engine.schedule_update_by == 'epoch':
        engine.lr.step()