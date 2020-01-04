import torch
import torch.optim as optim
from easydict import EasyDict as edict

from networks.dcgan_networks import Generator, Discriminator
from utils.utils import updateConfig
from models.utils import get_loss_criterion
from models.utils import get_scheduler

from models.meters.moving_avarage_meter import MovingAverageValueMeter
from models.meters.average_meter import AverageValueMeter

from models.losses.wgan_loss import WGAN_GP

import torch.nn.functional as F
import numpy as np

from functools import wraps


class BaseGAN():
    """
    Implementation of DCGAN
    """

    def __init__(self,
                 img_size=64,
                 nz=100,
                 lr_D=2e-4,
                 lr_G=1e-4,
                 beta1=0.5,
                 beta2=0.999,
                 use_gpu=True,
                 loss_criterion="BCE",
                 use_schedulerD=False,
                 use_schedulerG=False,
                 meterD=AverageValueMeter(),
                 meterG=AverageValueMeter(),
                 add_noise=True,
                 use_label_smoothing=False,
                 **kwargs):

        if "config" not in vars(self):
            self.config = edict()


        self.meterD = meterD
        self.meterG = meterG

        self.meterD2 = MovingAverageValueMeter(10)
        self.meterG2 = MovingAverageValueMeter(10)

        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
        self.config.use_label_smoothing = use_label_smoothing
        if self.config.use_label_smoothing:
            self.real_label = kwargs["smooth_label"]
        else:
            self.real_label = 1
        self.fake_label = 0    
        self.fake_real_label = 1

        self.epochs_trained = 0

        self.config.img_size = img_size
        self.config.nz = nz
        self.config.loss_criterion = loss_criterion
        self.config.add_noise = add_noise

        self.config.lr_D = float(lr_D)
        self.config.lr_G = float(lr_G)
        self.config.beta1 = beta1
        self.config.beta2 = beta2

        self.netG = self._get_netG()
        self.netD = self._get_netD()

        self.optimizerG = self._get_optimizerG()
        self.optimizerD = self._get_optimizerD()

        self.loss_criterion = get_loss_criterion(self.config.loss_criterion)
        # self.loss_criterion = WGAN_GP(self.netD, use_gp=False)

        # Schedulers configuration
        self.config.use_schedulerD = use_schedulerD
        if self.config.use_schedulerD:
            self.config.schedulerD_name = kwargs["schedulerD_name"]
            self.config.sD_c = kwargs["sD_c"]
            
            self.schedulerD = get_scheduler(self.config.schedulerD_name, self.optimizerD, **self.config.sD_c)

        self.config.use_schedulerG = use_schedulerG
        if self.config.use_schedulerG:
            self.config.schedulerG_name = kwargs["schedulerG_name"]
            self.config.sG_c = kwargs["sG_c"]
            
            self.schedulerG = get_scheduler(self.config.schedulerG_name, self.optimizerG, **self.config.sG_c)

        self._update_device()


    def _get_netG(self):
        pass


    def _get_netD(self):
        pass


    def _get_optimizerG(self):
        pass


    def _get_optimizerD(self):
        pass


    def load(self,
             path,
             in_state=None,
             load_netG=True,
             load_netD=True,
             load_optimD=True,
             load_optimG=True,
             load_config=True):

        in_state = torch.load(path)

        self._load_state_dict(in_state,
                              load_netG=load_netG,
                              load_netD=load_netD,
                              load_optimD=load_optimD,
                              load_optimG=load_optimG,
                              load_config=load_config)

        print(f"Loaded model: {path}")


    def load_netG_for_eval(self, path):
        pass
        

    def save(self, path, ext):

        state = self._get_state_dict()
        state["epochs_trained"] = self.epochs_trained

        save_name = str(self.epochs_trained) + ext

        save_path = "/".join([path, save_name])

        torch.save(state, save_path)
        print(f"Saved model: {save_path}")


    def _get_state_dict(self):
        """
        Get the model parameters
        """

        netG_state = self.netG.state_dict()
        netD_state = self.netD.state_dict()

        optimizerG_state = self.optimizerG.state_dict()
        optimizerD_state = self.optimizerD.state_dict()

        state = {
            "config": self.config,
            "netG": netG_state,
            "netD": netD_state,
            "optimizerG": optimizerG_state,
            "optimizerD": optimizerD_state,
            "epochs_trained": self.epochs_trained
        }

        return state


    def _load_state_dict(self,
                         in_state,
                         load_netG=True,
                         load_netD=True,
                         load_optimG=True,
                         load_optimD=True,
                         load_config=True,
                         train=True):

        if load_config:
            updateConfig(self.config, in_state["config"])
            self.loss_criterion = get_loss_criterion(self.config.loss_criterion)

        if load_netG:
            self.netG = self._get_netG()
            self.netG.load_state_dict(in_state["netG"])
            if train:
                self.netG.train()
            else:
                self.netG.eval()

        if load_netD:
            self.netD = self._get_netD()
            self.netD.load_state_dict(in_state["netD"])
            if train:
                self.netD.train()
            else:
                self.netD.eval()

        if load_optimD:
            self.optimizerD.load_state_dict(in_state["optimizerD"])

        if load_optimG:
            self.optimizerG.load_state_dict(in_state["optimizerG"])

        self.epochs_trained = in_state["epochs_trained"]

        self._update_device(in_state)



    # def _update_config(self, config):
    #     updateConfig(self.config, config)
    #     self._update_device()


    def _update_device(self, in_state=None):

        self.netD.to(self.device)
        self.netG.to(self.device)

        self.optimizerG = self._get_optimizerG()
        self.optimizerD = self._get_optimizerD()

        self.optimizerD.zero_grad()
        self.optimizerG.zero_grad()


    def generate_images(self, sample_size=1):
        pass


    def generate_fixed_noise(self, sample_size=1):
       pass


    def reset_meters(self):
        self.meterD.reset()
        self.meterD.reset()

        self.meterG2.reset()
        self.meterG2.reset()


    def _train_step(self, input_batch):
        raise NotImplementedError


   
    def scheduler(self, errD, errG):
        # Schedulers updates
        if self.config.use_schedulerD:
            self.schedulerD.step(errD)
        if self.config.use_schedulerG:
            self.schedulerG.step(errG)


    def meter(self, errD, errG):
        # Apply meters to the losses
        self.meterD.add(errD)
        self.meterG.add(errG)

        self.meterD2.add(errD)
        self.meterG2.add(errG)


    def optimize_generator(self):
        raise NotImplementedError


    def optimize_discriminator(self):
        raise NotImplementedError






    