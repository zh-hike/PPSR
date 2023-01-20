import os
import paddle


def log_train_info(engine, epoch_id, iter_id):
    batch_cost = engine.time_info['batch_cost'].info
    read_cost = engine.time_info['read_cost'].info
    loss = engine.train_loss_info.info
    total_iter_num = engine.cfg['Global']['epochs'] * engine.step_per_epoch
    remainder_iter_num = total_iter_num - epoch_id * engine.step_per_epoch - iter_id
    train_estimate_s = int(remainder_iter_num * engine.time_info['batch_cost'].avg)
    ips = int(engine.cfg['Data']['Train']['DataLoader']['batch_sampler']['batch_size'] * paddle.distributed.get_world_size() // engine.time_info['batch_cost'].avg)
    eta_day = train_estimate_s // 86400
    eta_h = (train_estimate_s % 86400) // 3600
    eta_m = (train_estimate_s % 3600) // 60
    eta_s = train_estimate_s % 60
    train_estimate_time = f"{eta_day}d {eta_h}:{eta_m}:{eta_s}"
    lr_info = "%.05f" % engine.opt.get_lr()
    info = f"Train: Epoch: [{epoch_id}/{engine.cfg['Global']['epochs']}]  Step: [{iter_id}/{engine.step_per_epoch}]  lr: {lr_info}  {batch_cost} {read_cost} {loss} ips: {ips}/s  eta: {train_estimate_time}"

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