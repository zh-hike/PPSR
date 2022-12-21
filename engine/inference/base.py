from .util import split_image, concat_image
import os
from PIL import Image
import paddle
from tqdm import tqdm


def inference_epoch_base(save_path, inference_model_path, test_path, img_size):
    os.makedirs(save_path, exist_ok=True)
    model = paddle.jit.load(inference_model_path)
    for filename in tqdm(os.listdir(test_path)):
        abspath = os.path.join(test_path, filename)
        if filename.split('.')[-1] in ['jpg', 'png']:
            img = Image.open(abspath)
            batch_img, params = split_image(img, img_size[1:])
            pred = model(batch_img)
            new_img = concat_image(pred, *params)
            new_img.save(os.path.join(save_path, filename))