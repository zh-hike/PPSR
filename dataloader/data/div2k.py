from ..base import BaseDataset


class DIV2K(BaseDataset):
    def __init__(self,
                 data_root,
                 index_file,
                 ops=None):
        super(DIV2K, self).__init__(ops=ops)
        super()._load_anno(index_file, data_root)