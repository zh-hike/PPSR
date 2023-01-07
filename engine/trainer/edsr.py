import time


def train_epoch_edsr(engine, epoch_id, iter_start=0):
    engine.info_reset()
    start_time = time.time()
    step_per_epoch = engine.cfg['Global']['step_per_epoch']
    engine.model.eval()
    engine.model.show_model()
    engine.model.align()
    assert 1==0