import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

import argparse
from engine.inference.base import inference_epoch_base
from utils.reader import read_yaml
from utils.set_config import recursion_set_config

def parse_args():
    parse = argparse.ArgumentParser("train ppsr")
    parse.add_argument('-c', '--config', required=True, help="参数文件(.yaml)所在位置")
    parse.add_argument("-o", "--override", action="append", default=[], help="增加附加的参数")
    args = parse.parse_args()
    cfg = read_yaml(args.config)
    for c in args.override:
        key, value = c.split("=")
        recursion_set_config(cfg, key.split('.'), value)
    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    save_path = os.path.join(cfg['Global'].get('output_dir', './output'), cfg['Arch']['name'], 'Img')
    inference_model_path = os.path.join(cfg['Global'].get('output_dir', './output'), cfg['Arch']['name'], 'inference', cfg['Arch']['name'])
    img_size = cfg['Global']['img_size'][1:]
    scale = cfg['Global'].get('scale', 1)
    rgb_range = cfg['Global'].get('rgb_range', 1)
    inference_epoch_base(save_path, inference_model_path, cfg['Data']['Test']['path'], img_size, scale=scale, rgb_range=rgb_range)