import time
from .util import log_train_info, save_checkoutpoints
import paddle
import os
from visualdl import LogWriter

def train_epoch_common(engine, epoch_id, iter_start=1):
    start_time = time.time()
    engine.time_info['batch_cost'].reset()
    engine.time_info['read_cost'].reset()
    step_per_epoch = engine.cfg['Global']['step_per_epoch']
    engine.train_loss_info.reset()
    train_dataloader_iter = iter(engine.train_dl)

    for iter_id in range(iter_start, step_per_epoch+1):
        try:
            batch = next(train_dataloader_iter)
        except:
            train_dataloader_iter = iter(engine.train_dl)
            batch = next(train_dataloader_iter)

        inputs, targets = batch
        engine.time_info['read_cost'].update(time.time() - start_time)

        if engine.amp:
            with paddle.amp.auto_cast(level=engine.amp_level):
                pred = engine.model(inputs)
                loss_dict = engine.train_loss_func(pred, targets)
        else:
            pred = engine.model(inputs)
            loss_dict = engine.train_loss_func(pred, targets)

        loss = loss_dict['loss']
        engine.opt.clear_grad()
        if engine.amp:
            scaled = engine.grad_scaler.scale(loss)
            scaled.backward()
            engine.grad_scaler.minimize(engine.opt, scaled)
        else:
            loss.backward()
            engine.opt.step()

        engine.train_loss_info.update(loss_dict)
        if engine.schedule_update_by == 'step':
            engine.lr.step()

        if iter_id % engine.cfg['Global']['print_batch_step'] == 0:
            log_train_info(engine, epoch_id, iter_id)
            engine.info_reset()

        engine.time_info['batch_cost'].update(time.time() - start_time)
        start_time = time.time()


    if engine.schedule_update_by == 'epoch':
        engine.lr.step()