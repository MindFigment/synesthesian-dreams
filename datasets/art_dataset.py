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

    def __init__(self, data_roots, transforms_=None):
        """

        Args:

        """
        self.data_roots = data_roots
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
        try:
            img = Image.open(img_path).convert("RGB")
        except IOError:
            print(f"Error processing {img_path}")
            exit(-1)

        if self.transform:
            img = self.transform(img)

        return img

    
    def _init_dataset(self):
        img_list =  []
        for data_root in self.data_roots:
            img_list += glob.glob(os.path.join(data_root, "**", "*.jpg"))

        return img_list


if __name__ == "__main__":

    img_size = 64

    data_root = "./data/images/images"
    dataset = ArtDataset(data_root, transforms_= [
                                        transforms.Resize(img_size),
                                        transforms.CenterCrop(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    