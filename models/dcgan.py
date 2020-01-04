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
                 ngf=64,
                 ndf=64,
                 nc=3,
                 use_spectral_norm_D=False,
                 use_spectral_norm_G=False,
                 **kwargs):

        if "config" not in vars(self):
            self.config = edict()

        # self.config.nz = nz
        self.config.ndf = ndf
        self.config.ngf = ngf
        self.config.nc = nc
        self.config.use_spectral_norm_D = use_spectral_norm_D
        self.config.use_spectral_norm_G = use_spectral_norm_G

        BaseGAN.__init__(self, **kwargs)


    def _get_netG(self):
        netG = Generator(self.config.img_size, self.config.nz, self.config.ngf, self.config.nc, self.config.use_spectral_norm_G)
        return netG


    def _get_netD(self):
        netD = Discriminator(self.config.img_size, self.config.ndf, self.config.nc, self.config.use_spectral_norm_D)
        return netD


    def _get_optimizerG(self):
        return optim.Adam(self.netG.parameters(), lr=self.config.lr_G, betas=(self.config.beta1, self.config.beta2))


    def _get_optimizerD(self):
        return optim.Adam(self.netD.parameters(), lr=self.config.lr_D, betas=(self.config.beta1, self.config.beta2))


    def generate_images(self, sample_size=1):

        fixed_noise = self.generate_fixed_noise(sample_size)

        with torch.no_grad():
            fake = self.netG(fixed_noise).detach().cpu()

        return fake

    
    def generate_fixed_noise(self, sample_size=1):
        fixed_noise = torch.randn(sample_size, self.config.nz, 1, 1, device=self.device)
        return fixed_noise


    def generater_fixed_images(self, fixed_noise):
        with torch.no_grad():
            fake = self.netG(fixed_noise).detach().cpu()

        return fake


    def load_netG_for_eval(self, path):
        # Load Model
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint["config"]
        netG_params = {
            "img_size": config.img_size,
            "nz": config.nz,
            "ngf": config.ngf,
            "nc": config.nc
        }
        # Create and load generator
        self.netG = Generator(**netG_params).to(self.device)
        self.netG.load_state_dict(checkpoint["netG"])
        # print(netG)
        self.netG.eval()


    def _train_step(self, real_samples):

        batch_size = real_samples.size(0)
        noise = torch.randn(batch_size, self.config.nz, 1, 1, device=self.device)

        errD, D_x, D_G_z1 = self.optimize_discriminator(real_samples, noise, batch_size)
        errG, D_G_z2 = self.optimize_generator(noise, batch_size)

        self.scheduler(errD, errG)
        self.meter(errD, errG)

        return errD, errG, D_x, D_G_z1, D_G_z2
        # return self.meterD.value(), self.meterG.value(), D_x, D_G_z1, D_G_z2
        # return self.meterD2.value(), self.meterG2.value(), D_x, D_G_z1, D_G_z2


    def optimize_generator(self, noise, batch_size):
        fake_samples = self.netG(noise)

        label = torch.full((batch_size,), self.real_label, device=self.device)
        output = self.netD(fake_samples).view(-1)
        errG = self.loss_criterion(output, label)
        
        self.netG.zero_grad()
        errG.backward()
        self.optimizerG.step()

        D_G_z2 = output.mean().item()

        return errG.item(), D_G_z2


    def optimize_discriminator(self, real_samples, noise, batch_size):

        self.netD.zero_grad()

        fake_samples = self.netG(noise)
        real_samples = real_samples.to(self.device)
        label = torch.full((batch_size,), self.real_label, device=self.device)
        output1 = self.netD(real_samples).view(-1)
        errD_real = self.loss_criterion(output1, label)

        errD_real.backward()

        label.fill_(self.fake_label)
        output2 = self.netD(fake_samples.detach()).view(-1)
        errD_fake = self.loss_criterion(output2, label)

        errD_fake.backward()

        self.optimizerD.step()

        D_x = output1.mean().item()
        D_G_z1 = output2.mean().item()
        errD = (errD_real + errD_fake) / 2

        return errD.item(), D_x, D_G_z1


        








    