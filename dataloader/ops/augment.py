from paddle.vision.transforms import functional as F
from paddle.vision.transforms import BaseTransform
from PIL import Image
import numpy as np
import random


def _get_img_size(img):
    assert isinstance(img, Image.Image)
    if isinstance(img, Image.Image):
        return img.size


class RandomCrop(BaseTransform):
    def __init__(self, 
                 size,
                 padding=None,
                 pad_if_needed=False,
                 fill=0,
                 padding_mode='reflect',
                 keys=None,
                 **kwargs):
        super(RandomCrop, self).__init__(keys)
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)
        self.padding = padding
        self.fill = fill
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode


    def _get_param(self, img, output_size):
        w, h = _get_img_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h-th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


    def __call__(self, img1, img2=None):
        if self.padding is not None:
            img1 = F.pad(img1, self.padding, self.fill, self.padding_mode)
            if img2:
                img2 = F.pad(img2, self.padding, self.fill, self.padding_mode)

        w, h = _get_img_size(img1)
        if self.pad_if_needed and w < self.size[1]:
            img1 = F.pad(img1, (self.size[1] - w, 0), self.fill, self.padding_mode)
            if img2:
                img2 = F.pad(img2, (self.size[1] - w, 0), self.fill, self.padding_mode)

        if self.pad_if_needed and h < self.size[0]:
            img1 = F.pad(img1, (0, self.size[0] - h), self.fill, self.padding_mode)
            if img2:
                img2 = F.pad(img2, (0, self.size[0] - h), self.fill, self.padding_mode)
        
        i, j, h, w = self._get_param(img1, self.size)
        if img2:
            return F.crop(img1, i, j, h, w), F.crop(img2, i, j, h, w)
        else:
            return F.crop(img1, i, j, h, w)


class ToTensor(BaseTransform):
    def __init__(self, data_format="CHW", keys=None):
        super(ToTensor, self).__init__(keys)
        self.data_format = data_format

    def __call__(self, img1, img2=None):
        if img2:
            return F.to_tensor(img1, data_format=self.data_format), F.to_tensor(img2, data_format=self.data_format)
        else:
            return F.to_tensor(img1, data_format=self.data_format)