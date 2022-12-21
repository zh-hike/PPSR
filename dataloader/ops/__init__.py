from .augment import RandomCrop, ToTensor
from PIL import Image


def transforms(img1, img2=None, ops=None):
    if ops is None:
        if img2 is not None:
            return img1, img2
        else:
            return img1
    assert isinstance(img1, Image.Image), f"img 仅支持PIL.Image格式，不支持{type(img1)}格式"
    assert isinstance(img2, (Image.Image, None))
    for op in ops:
        ops_name = list(op.keys())[0]
        if img2 is not None:
            img1, img2 = eval(ops_name)(**op[ops_name])(img1, img2)
        else:
            img1 = eval(ops_name)(**op[ops_name])(img1)
    if img2 is not None:
        return img1, img2
    else:
        return img1