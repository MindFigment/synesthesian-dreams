from models.dcgan import DCGAN
from logger.utils import log_train_step_stats
from datasets.art_dataset import ArtDataset

from easydict import EasyDict as edict

import torch
from torchvision import transforms, utils
from trainers.base_gan_trainer import BaseGANTrainer

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


    def train(self):

        if self.t_c.load:
            self.model.load(path=self.t_c.load_model,
                            load_netD=self.t_c.load_netD,
                            load_netG=self.t_c.load_netG,
                            load_optimD=self.t_c.load_optimD,
                            load_optimG=self.t_c.load_optimG)
        
        self.start = self.model.epochs_trained
        self.end = self.t_c.epochs  + self.start
            

        for epoch in range(self.start, self.end):
            dataloader = self._get_dataloader()
            self._train_one_epoch(dataloader, epoch)
