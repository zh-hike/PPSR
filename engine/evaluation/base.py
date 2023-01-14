import paddle
import time
from PIL import Image
from dataloader.ops import split_image, concat_image, ToTensor
from ..util import prepare_before_inference
from utils.util import read_img
from tqdm import tqdm


@paddle.no_grad()
def eval_epoch_base(engine, **kwargs):
    start_time = time.time()
    engine.time_info['read_cost'].reset()
    engine.time_info['batch_cost'].reset()
    engine.eval_metric_info.reset()
    engine.eval_loss_info.reset()
    clean_imgs = engine.eval_dl.dataset.clean_imgs
    noise_imgs = engine.eval_dl.dataset.noise_imgs
    scale = engine.scale
    to_tensor = ToTensor(rgb_range=engine.rgb_range)
    size = engine.cfg['Global']['img_size'][1:]
    rgb_range = engine.rgb_range
    for batch, (inputs, targets) in enumerate(tqdm(zip(noise_imgs, clean_imgs), total=len(noise_imgs), disable=engine.eval_bar_disable, ncols=100)):
        engine.time_info['read_cost'].update(time.time() - start_time)
        
        noise_img = read_img(inputs)
        target_img = read_img(targets)
        targets = to_tensor(target_img)
        batch_noise_imgs, params = prepare_before_inference(noise_img, size=size, rgb_range=rgb_range, scale=scale)

        batch_noise_imgs = batch_noise_imgs
        pred = engine.model(batch_noise_imgs)

        pred = concat_image(pred, *params, rgb_range=rgb_range, need_to_pil=False)
        if getattr(engine, "eval_loss_func", False):
            loss = engine.eval_loss_func(pred.unsqueeze(0), targets.unsqueeze(0))
            engine.eval_loss_info.update(loss)

        metric_result = engine.eval_metric_func(pred.unsqueeze(0), targets.unsqueeze(0))
        engine.eval_metric_info.update(metric_result)
        engine.time_info['batch_cost'].update(time.time() - start_time)
        start_time = time.time()