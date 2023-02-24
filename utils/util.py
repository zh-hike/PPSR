import paddle
from PIL import Image
import os
import io
import numpy as np
import random
import re


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


def format_dict(s: dict, level=0):

    def format_dict_inter(s: dict, level=0):
        if not isinstance(s, dict):
            if s is None:
                return 'null'
            if isinstance(s, list):
                m = []
                if not isinstance(s[0], dict):
                    m.append(str(s))
                    return ''.join(m)
                else:
                    for item in s:
                        m.append(format_dict_inter(item, level+1))

                return '\n' + ''.join(m)
            return str(s)
        m = []
        for key, value in s.items():
            s = format_dict_inter(value, level+1)
            v = '\n' if isinstance(value, dict) else ''
            m.append("    "*level + f"{key}: {v}{s}")
        split = '-'*80+'\n' if level==0 else ''
        result = f'\n{split}'.join(m) + '\n'
        return re.sub('(\n+)', '\n', result)
    result ="\n" + "="*100 + "\n"
    result += format_dict_inter(s, level)
    result = result + "="*100 + "\n"
    return result