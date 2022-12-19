from paddle.vision.transforms.transforms import ToTensor, RandomCrop
from PIL import Image


def transforms(img, ops):
    if ops is None:
        return img
    assert isinstance(img, Image.Image), f"img 仅支持PIL.Image格式，不支持{type(img)}格式"
    for op in ops:
        ops_name = list(op.keys())[0]
        img = eval(ops_name)(**op[ops_name])(img)
    
    return img

