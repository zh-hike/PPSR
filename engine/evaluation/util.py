import os
import paddle

def log_eval_info(engine):
    batch_cost = engine.time_info['batch_cost'].info
    read_cost = engine.time_info['read_cost'].info
    metric_result = engine.eval_metric_info.info
    loss = ""
    best_metric = "best_metric: %.5f " % engine.best_metric_value
    if getattr(engine, "eval_loss_info"):
        loss = engine.eval_loss_info.info

    info = f"Eval: {batch_cost}  {read_cost} {loss} {best_metric} {metric_result}\n"

    engine.eval_logger.info(info)