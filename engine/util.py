import paddle
import time
from PIL import Image
from dataloader.ops import split_image, concat_image, ToTensor


def prepare_before_inference(img, size, rgb_range, scale=1):
    patch_imgs, params = split_image(img, size=size, rgb_range=rgb_range)
    nh, nw, h, w = params
    params = [nh, nw, h*scale, w*scale]
    return patch_imgs, params
