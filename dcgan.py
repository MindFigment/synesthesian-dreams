from .networks.dcgan_networks import Generator, Discriminator
from .utils.config import updateConfig
from .losses.base_losses import get_loss_criterion

import torch
import torch.optim as optim

class DCGAN():
    """
    Implementation of DCGAN
    """

    def __init__(self,
                 lr,
                 beta1,
                 beta2=0.999,
                 latent_vector_size=100,
                 feature_maps_size=64,
                 channels_num=3,
                 use_gpu=True,
                 loss_criterion="BCE"):

        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.real_label = 1
        self.fake_label = 0

        self.config = {}

        self.config.latent_vector_size = latent_vector_size
        self.config.feature_maps_size = feature_maps_size
        self.config.channels_num = channels_num
        self.config.loss_criterion = loss_criterion

        self.config.lr = lr
        self.config.beta1 = beta1
        self.config.beta2 = beta2

        self.netG = self._getNetG()
        self.netD = self._getNetD()

        self.optimizerG = self._getOptimizerG()
        self.optimizerD = self._getOptimizerD()

        self.loss_criterion = get_loss_criterion(self.config.loss_criterion)

        self._update_device()


    def _getNetG(self):
        netG = Generator()
        return netG


    def _getNetD(self):
        netD = Discriminator()
        return netD


    def _getOptimizerG(self):
        return optim.Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))


    def _getOptimizerD(self):
        return optim.Adam(self.netD.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))


    def load(self,
             path="",
             in_state=None,
             loadG=True,
             loadD=True,
             loadConfig=True):

        in_state = torch.load(path)
        self._load_state_dict(in_state,
                              loadG=loadG,
                              loadD=loadD,
                              loadConfig=loadConfig)


    def save(self, path):
        torch.save(self._get_state_dict(), path)


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
            "optimizerD": optimizerD_state
        }

        return state


    def _load_state_dict(self,
                         in_state,
                         loadG=True,
                         loadD=True,
                         loadConfig=True,
                         train=True):

        if loadConfig:
            updateConfig(self.config, in_state["config"])
            self.loss_criterion = get_loss_criterion(self.config.loss_criterion)

        if loadG:
            self.netG = self._getNetG()
            if train:
                self.netG.train()
            else:
                self.netG.eval()

        if loadD:
            self.netD = self._getNetD()
            if train:
                self.netD.train()
            else:
                self.netD.eval()

        self._update_device()



    def _update_config(self, config):
        updateConfig(self.config, config)
        self._update_device()


    def _update_device(self):

        self.netD.to(self.device)
        self.netG.to(self.device)

        self.optimizerD = self.optimizerD()
        self.optimizerG = self.optimizerG()

        self.optimizerD.zero_grad()
        self.optimizerG.zero_grad()


    def train_step(self, input_batch):

        # Perform one step of learning
        ##############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ##############################
        # Train with all-real batch
        self.netD.zero_grad()
        # Format batch
        real_input = input_batch.to(self.device)
        batch_size = real_input.size(0)
        label = torch.full((batch_size,), self.real_label, device=self.device)
        # Forward pass real batch through D
        output = self.netD(real_input).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.loss_criterion(output, label)
        # calculate gradeints for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, self.config.latent_vector_size, 1, 1, device=self.device)
        # Generate fake image batch with G
        fake = self.netG(noise)
        label.fill_(self.fake_label)
        # Classify all fake batch with D
        output = self.netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.loss_criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizerD.step()

        ###########################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.netG.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.loss_criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.optimizerG.step()
        







    