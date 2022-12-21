import paddle
import math
import numpy as np
from PIL import Image
from paddle.vision.transforms import functional as F


def to_pil(tensor) -> Image.Image:
    assert isinstance(tensor, paddle.Tensor)
    tensor = paddle.clip(tensor, 0, 1)
    d = tensor.numpy()
    d = np.uint8(d*255).transpose(1,2,0)
    img = Image.fromarray(d)
    return img


def split_image(img, size, padding_mode='reflect'):
    assert isinstance(size, (int, tuple, list))
    assert isinstance(img, Image.Image)
    if isinstance(size, int):
        ph, pw = size, size
    else:
        ph, pw = size
    w, h = img.size
    pad_w, pad_h = 0, 0
    nh, nw = h//ph, w // pw
    if w % pw != 0:
        pad_w = math.ceil(w/pw)*pw - w
        nw += 1
    if h % ph != 0:
        pad_h = math.ceil(h/ph)*ph - h
        nh += 1
    img = F.pad(img, (0, 0, pad_w, pad_h), padding_mode=padding_mode)
    
    it = F.to_tensor(img)
    im = paddle.split(it, nh, axis=1)
    im = paddle.stack(im, axis=0)
    im = paddle.split(im, nw, axis=-1)
    im = paddle.concat(im, axis=0)
    return im, (nh, nw, h, w)


def concat_image(img, nh, nw, h, w) -> Image.Image:
    assert isinstance(img, paddle.Tensor)
    img = paddle.split(img, nw, axis=0)
    img = paddle.concat(img, axis=-1)
    
    img = paddle.unbind(img, axis=0)
    img = paddle.concat(img, axis=1)
    img = img[:, :h, :w]
    
    return to_pil(img)