from PIL import Image
from paddle.io import Dataset
from .ops import transforms
import os
from utils.util import read_img


class BaseDataset(Dataset):
    def __init__(self, ops):
        self.clean_imgs = []
        self.noise_imgs = []
        self.ops = ops

        
    def _load_anno(self, index_file, data_root):
        file = open(index_file, 'r')
        for line in file.readlines():
            noise_img, cls_img = line.strip().split()
            noise_img = os.path.join(data_root, noise_img.strip())
            cls_img = os.path.join(data_root, cls_img.strip())
            self.noise_imgs.append(noise_img)
            self.clean_imgs.append(cls_img)
        file.close()

    def data_expand(self, data_expand=None):
        if data_expand is not None:
            self.clean_imgs = self.clean_imgs * data_expand
            self.noise_imgs = self.noise_imgs * data_expand

    def __getitem__(self, idx):
        
        clean_img = self.clean_imgs[idx]
        noise_img = self.noise_imgs[idx]

        clean_img = read_img(clean_img)
        noise_img = read_img(noise_img)

        noise_img, clean_img = transforms(noise_img, clean_img, self.ops)
        
        return noise_img, clean_img

    def __len__(self):
        return len(self.clean_imgs)