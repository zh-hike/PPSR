from ..base import BaseDataset


class ValDataset(BaseDataset):
    def __init__(self, data_root, index_file):
        super(ValDataset, self).__init__(ops=None)
        super()._load_anno(index_file, data_root)
        
    def __getitem__(self, idx):
        return self.noise_imgs[idx], self.clean_imgs[idx]