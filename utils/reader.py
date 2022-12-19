import yaml
import os

def read_yaml(path):
    assert os.path.exists(path), f"参数文件{path}不存在"
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg