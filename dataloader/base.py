from PIL import Image
from paddle.io import Dataset
from .ops import transforms


class BaseDataset(Dataset):
    def __init__(self, ops):
        self.clean_imgs = []
        self.noise_imgs = []
        self.ops = ops


    def __getitem__(self, idx):
        
        clean_img = self.clean_imgs[idx]
        noise_img = self.noise_imgs[idx]
        clean_img = Image.open(clean_img)
        noise_img = Image.open(noise_img)

        noise_img, clean_img = transforms(noise_img, clean_img, self.ops)
        return noise_img, clean_img

    def __len__(self):
        return len(self.clean_imgs)