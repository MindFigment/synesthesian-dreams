import os
import glob
import itertools as it
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ArtDataset(Dataset):
    """ Dataset of artworks """

    def __init__(self, data_root, transforms_=None):
        """

        Args:

        """
        self.data_root = data_root
        self.transform = transforms.Compose(transforms_)
        self.images =  self._init_dataset()


    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.images)


    def __getitem__(self, idx):
        """ Generates one sample of data """
        # Select image
        img_path = self.images[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # print(img_path)

        if self.transform:
            img = self.transform(img)

        return img

    
    def _init_dataset(self):
        image_list = glob.glob(os.path.join(self.data_root, "**", "*.jpg"))
        return image_list


if __name__ == "__main__":

    img_size = 64

    data_root = "./data/images/images"
    dataset = ArtDataset(data_root, transforms_= [
                                        transforms.Resize(img_size),
                                        transforms.CenterCrop(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    