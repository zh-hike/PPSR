import paddle


def quantize(img, rgb_range=1):
    img = paddle.clip(255/rgb_range*img, 0, 255).astype('float32')
    return img