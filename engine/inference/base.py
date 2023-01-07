from dataloader.ops import concat_image
from ..util import prepare_before_inference
import os
from PIL import Image
import paddle
from tqdm import tqdm


def inference_epoch_base(save_path, inference_model_path, test_path, img_size, scale=1, rgb_range=1):
    os.makedirs(save_path, exist_ok=True)
    model = paddle.jit.load(inference_model_path)

    for filename in tqdm(os.listdir(test_path)):
        abspath = os.path.join(test_path, filename)
        if filename.split('.')[-1] in ['jpg', 'png']:
            img = Image.open(abspath)
            batch_img, params = prepare_before_inference(img, size=img_size, rgb_range=rgb_range, scale=scale)
            pred = model(batch_img)
            new_img = concat_image(pred, *params, rgb_range=rgb_range)
            new_img.save(os.path.join(save_path, filename))

    print(f"Inference succeeded! The imgs has been saved in {save_path}")