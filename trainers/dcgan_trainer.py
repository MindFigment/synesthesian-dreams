from models.dcgan import DCGAN
from logger.utils import log_train_step_stats
from datasets.art_dataset import ArtDataset

from easydict import EasyDict as edict
from pprint import pprint

import torch
from torchvision import transforms, utils
from trainers.base_gan_trainer import BaseGANTrainer

import numpy as np


class DCGANTrainer(BaseGANTrainer):
    """
    .
    """

    def __init__(self,
                 t_c,
                 m_c
                 ):

        BaseGANTrainer.__init__(self, t_c, m_c)


    def init_model(self):
        self.model = DCGAN(img_size=self.t_c.img_size, **self.m_c)



