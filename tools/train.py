import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

import argparse
from engine.engine import Engine
from utils.reader import read_yaml

def parse_args():
    parse = argparse.ArgumentParser("train ppsr")
    parse.add_argument('-c', '--config', required=True, help="参数文件(.yaml)所在位置")
    
    args = parse.parse_args()
    return read_yaml(args.config)

if __name__ == "__main__":
    cfg = parse_args()
    engine = Engine(cfg)
    engine.train()
    