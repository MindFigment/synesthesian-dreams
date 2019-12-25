import torch
import torch.optim as optim
from easydict import EasyDict as edict

from models.base_gan import BaseGAN
from networks.dcgan_networks import Generator, Discriminator
from utils.utils import updateConfig
from models.utils import get_loss_criterion

class DCGAN(BaseGAN):
    """
    Implementation of DCGAN
    """

    def __init__(self,
                 ngf,
                 ndf,
                 nc,
                 **kwargs):

        if "config" not in vars(self):
            self.config = edict()

        # self.config.nz = nz
        self.config.ndf = ndf
        self.config.ngf = ngf
        self.config.nc = nc

        BaseGAN.__init__(self, **kwargs)


    def _get_netG(self):
        netG = Generator(self.config.img_size, self.config.nz, self.config.ngf, self.config.nc)
        return netG


    def _get_netD(self):
        netD = Discriminator(self.config.img_size, self.config.ndf, self.config.nc)
        return netD


    def _get_optimizerG(self):
        return optim.Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))


    def _get_optimizerD(self):
        return optim.Adam(self.netD.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))


        







    