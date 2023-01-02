from ..base import BaseDataset
from PIL import Image
import os


class WaterMark(BaseDataset):
    def __init__(self,
                 data_root,
                 index_file,
                 ops=None):
        
        super(WaterMark, self).__init__(ops=ops)
        super()._load_anno(index_file, data_root)