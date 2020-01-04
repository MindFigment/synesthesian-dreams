from models.msggan import MSGGAN
from logger.utils import log_train_step_stats
from datasets.art_dataset import ArtDataset

from easydict import EasyDict as edict
from pprint import pprint

import torch
from torchvision import transforms, utils
from trainers.base_gan_trainer import BaseGANTrainer
from torchsummary import summary

import numpy as np


class MSGGANTrainer(BaseGANTrainer):
    """
    .
    """

    def __init__(self,
                 t_c,
                 m_c
                 ):

        BaseGANTrainer.__init__(self, t_c, m_c)


    def init_model(self):
        print(self.m_c)
        self.model = MSGGAN(img_size=self.t_c.img_size, **self.m_c)
        # print(self.model.netG)
        # print(self.model.netD)

        print("MODEL G")
        summary(self.model.netG, input_size=(self.m_c.nz,))
        print("MODEL_D")
        # summary(self.model.netD, input_size=(3, self.t_c.img_size, self.t_c.img_size))


