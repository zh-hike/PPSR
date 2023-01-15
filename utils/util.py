import paddle
from PIL import Image
import os
import io
import numpy as np
import random


def quantize(img, rgb_range=1):
    img = paddle.clip(255/rgb_range*img, 0, 255).astype('float32')
    return img

def read_img(file, return_pil=True):
    img = Image.open(file)
    return img

    # assert os.path.exists(file), f"{file} is not exists!!"
    # with open(file, 'rb') as f:
    #     data = io.BytesIO(f.read())
    #     img = Image.open(data)
    #     img_array = np.asarray(img)

    # img = Image.fromarray(img_array) if return_pil else img_array
    # return img

def set_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)