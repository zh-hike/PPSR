from ..base import BaseDataset
from PIL import Image
import os


class WaterMark(BaseDataset):
    def __init__(self,
                 clean_data,
                 noise_data,
                 clean_ops=None,
                 noise_ops=None):

        super(WaterMark, self).__init__(clean_ops=clean_ops,
                                        noise_ops=noise_ops)
        self.clean_imgs = []
        self.noise_imgs = []
        for noise_img_file in os.listdir(noise_data):
            self.noise_imgs.append(os.path.join(noise_data, noise_img_file))
            clean_img_file = noise_img_file.split('_')[0]+'.jpg'
            self.clean_imgs.append(os.path.join(clean_data, clean_img_file))

        
