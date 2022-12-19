from PIL import Image
from paddle.io import Dataset
from .ops import transforms


class BaseDataset(Dataset):
    def __init__(self, clean_ops=None, noise_ops=None):
        self.clean_imgs = []
        self.noise_imgs = []
        self.clean_ops = clean_ops
        self.noise_ops = noise_ops


    def __getitem__(self, idx):
        
        clean_img = self.clean_imgs[idx]
        noise_img = self.noise_imgs[idx]
        clean_img = Image.open(clean_img)
        noise_img = Image.open(noise_img)

        noise_img = transforms(noise_img, self.noise_ops)
        clean_img = transforms(clean_img, self.clean_ops)
        return noise_img, clean_img

    def __len__(self):
        return len(self.clean_imgs)