import paddle
import time


@paddle.no_grad()
def eval_epoch_base(engine, **kwargs):
    start_time = time.time()
    engine.time_info['read_cost'].reset()
    engine.time_info['batch_cost'].reset()
    engine.eval_metric_info.reset()
    engine.eval_loss_info.reset()
    
    for batch, (inputs, targets) in enumerate(engine.eval_dl):
        engine.time_info['read_cost'].update(time.time() - start_time)
        pred = engine.model(inputs)
        if getattr(engine, "eval_loss_func", False):
            loss = engine.eval_loss_func(pred, targets)
            engine.eval_loss_info.update(loss)

        metric_result = engine.eval_metric_func(pred, targets)
        engine.eval_metric_info.update(metric_result)

        engine.time_info['batch_cost'].update(time.time() - start_time)
        start_time = time.time()