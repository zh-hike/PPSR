from utils import logger
import os
import paddle


def log_train_info(engine, epoch_id, iter_id):
    batch_cost = round(engine.time_info['batch_cost'].avg, 4)
    read_cost = round(engine.time_info['read_cost'].avg, 4)
    loss = round(engine.train_loss_info.avg, 4)
    train_estimate_s = engine.cfg['Global']['epochs']*engine.cfg['Global']['step_per_epoch']*batch_cost
    train_estimate_time = f"{int(train_estimate_s // 86400)} day {int((train_estimate_s%86400)//3600)} h"
    lr_info = "%.05f"%engine.opt.get_lr()
    info = f"""Train: Epoch: [{epoch_id}/{engine.cfg['Global']['epochs']}]  Step: [{iter_id}/{engine.cfg['Global']['step_per_epoch']}]  lr: {lr_info}  batch_cost: {batch_cost}s  read_cost: {read_cost}s   Loss: {loss}  run_time_estimate: {train_estimate_time}"""
    logger.info(info)


def save_checkoutpoints(engine, epoch_id, iter_id):
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoints = {'model': engine.model.state_dict(),
                   'optimizer': engine.opt.state_dict(),
                   'epoch_id': epoch_id,
                   'iter_id': iter_id,
                   }
    paddle.save(checkpoints, f'./checkpoints/{engine._name}.checkpoints')

    

