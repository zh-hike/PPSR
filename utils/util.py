import paddle
from PIL import Image
import os
import io


def quantize(img, rgb_range=1):
    img = paddle.clip(255/rgb_range*img, 0, 255).astype('float32')
    return img

def read_img(file):
    img = Image.open(file)
    return img

    # assert os.path.exists(file), f"{file} is not exists!!"
    # with open(file, 'rb') as f:
    #     data = io.BytesIO(f.read())
    #     img = Image.open(data)

    # return img