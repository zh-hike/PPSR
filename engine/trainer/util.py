import os
import paddle


def log_train_info(engine, epoch_id, iter_id):
    batch_cost = engine.time_info['batch_cost'].info
    read_cost = engine.time_info['read_cost'].info
    loss = engine.train_loss_info.info
    total_iter_num = engine.cfg['Global']['epochs'] * engine.cfg['Global']['step_per_epoch']
    remainder_iter_num = total_iter_num - epoch_id * engine.cfg['Global']['step_per_epoch'] - iter_id
    train_estimate_s = remainder_iter_num * engine.time_info['batch_cost'].avg
    train_estimate_time = f"{int(train_estimate_s // 86400)}day {int((train_estimate_s%86400)//3600)}h"
    lr_info = "%.05f" % engine.opt.get_lr()
    info = f"Train: Epoch: [{epoch_id}/{engine.cfg['Global']['epochs']}]  Step: [{iter_id}/{engine.cfg['Global']['step_per_epoch']}]  lr: {lr_info}  {batch_cost} {read_cost} {loss} run_time_estimate: {train_estimate_time}"

    engine.train_logger.info(info)


def save_checkoutpoints(engine, epoch_id, iter_id):
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoints = {'model': engine.model.state_dict(),
                   'optimizer': engine.opt.state_dict(),
                   'epoch_id': epoch_id,
                   'iter_id': iter_id,
                   }
    print(f"save ck epoch_{epoch_id}  iter_{iter_id}")
    paddle.save(checkpoints, f'./checkpoints/{engine._name}.checkpoints')