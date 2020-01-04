import torch
import torch.optim as optim
from easydict import EasyDict as edict
import copy
import numpy as np

from models.base_gan import BaseGAN
from networks.msggan_networks import Generator, Discriminator
from utils.utils import updateConfig
from models.utils import get_loss_criterion

from networks.weights_initializer import weights_init

import torch.nn.functional as F

from pprint import pprint



class MSGGAN(BaseGAN):
    """
    Implementation of MSG-GAN
    """

    def __init__(self,
                 depth=7, # Means 128 x 128 final ing size
                 nz=64,
                 ndf=64,
                 nc=3,
                 use_spectral_norm_D=False,
                 use_spectral_norm_G=False,
                 **kwargs):

        if "config" not in vars(self):
            self.config = edict()

        self.config.depth = depth
        # self.config.nz = nz
        self.config.ndf = ndf
        self.config.nc = nc
        self.config.use_spectral_norm_D = use_spectral_norm_D
        self.config.use_spectral_norm_G = use_spectral_norm_G

        BaseGAN.__init__(self, nz=nz, **kwargs)

        pprint(self.config)



    def _get_netG(self):
        netG = Generator(self.config.depth, self.config.nz, self.config.nc, self.config.use_spectral_norm_G)
        # netG.apply(weights_init)
        return netG


    def _get_netD(self):
        netD = Discriminator(self.config.depth, self.config.ndf, self.config.nc, self.config.use_spectral_norm_D)
        # netD.apply(weights_init)
        return netD


    def _get_optimizerG(self):
        return optim.Adam(self.netG.parameters(), lr=self.config.lr_G, betas=(self.config.beta1, self.config.beta2))


    def _get_optimizerD(self):
        return optim.Adam(self.netD.parameters(), lr=self.config.lr_D, betas=(self.config.beta1, self.config.beta2))


    def generate_images(self, sample_size=1):

        fixed_noise = self.generate_fixed_noise(sample_size)

        with torch.no_grad():
            fake = [ x.detach().cpu() for x in self.netG(fixed_noise) ]

        return fake


    def generater_fixed_images(self, fixed_noise):
        with torch.no_grad():
            fake = [ x.detach().cpu() for x in self.netG(fixed_noise) ]

        return fake

    
    def generate_fixed_noise(self, sample_size=1):
        fixed_noise = torch.randn(sample_size, self.config.nz, device=self.device)
        return fixed_noise


    def load_netG_for_eval(self, path):
        # Load Model
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint["config"]
        netG_params = {
            "depth": config.depth,
            "nz": config.nz,
            "nc": config.nc,
            "use_spectral_norm": False#config.use_spectral_norm_G,
        }
        # Create and load generator
        self.netG = Generator(**netG_params).to(self.device)
        self.netG.load_state_dict(checkpoint["netG"])
        # print(netG)
        self.netG.eval()


    def _train_step(self, input_batch):

        batch_size = input_batch.size(0)
        real_samples = [input_batch] + [F.avg_pool2d(input_batch, int(np.power(2, i))) for i in range(1, self.config.depth)]
        noise = torch.randn(batch_size, self.config.nz, device=self.device)

        errD, D_x, D_G_z1 = self.optimize_discriminator(real_samples, noise, batch_size)
        errG, D_G_z2 = self.optimize_generator(noise, batch_size)

        self.scheduler(errD, errG)
        self.meter(errD, errG)

        return errD, errG, D_x, D_G_z1, D_G_z2


    def optimize_generator(self, noise, batch_size):
        fake_samples = self.netG(noise)

        label = torch.full((batch_size,), self.fake_real_label, device=self.device)
        output = self.netD(fake_samples).view(-1)
        errG = self.loss_criterion(output, label)
        
        self.netG.zero_grad()
        errG.backward()
        self.optimizerG.step()

        # D_G_z2 = output.mean().item()
        D_G_z2 = torch.sigmoid(output.detach()).mean().item()

        return errG.item(), D_G_z2


    def optimize_discriminator(self, real_samples, noise, batch_size):

        self.netD.zero_grad()

        fake_samples = self.netG(noise)
        fake_samples = list(map(lambda x: x.detach(), fake_samples))
        real_samples = list(map(lambda x: x.to(self.device), real_samples))
        label = torch.full((batch_size,), self.real_label, device=self.device)
        output1 = self.netD(real_samples).view(-1)
        errD_real = self.loss_criterion(output1, label)

        errD_real.backward()

        label.fill_(self.fake_label)
        # label2 = torch.full((batch_size,), self.fake_label, device=self.device)
        output2 = self.netD(fake_samples).view(-1)
        errD_fake = self.loss_criterion(output2, label)

        errD_fake.backward()

        self.optimizerD.step()

        # D_x = output1.mean().item()
        D_x = torch.sigmoid(output1.detach()).mean().item()
        # D_G_z1 = output2.mean().item()
        D_G_z1 = torch.sigmoid(output2.detach()).mean().item()
        errD = (errD_real + errD_fake) / 2

        return errD.item(), D_x, D_G_z1






    