import torch
import torch.optim as optim
from easydict import EasyDict as edict

from networks.dcgan_networks import Generator, Discriminator
from utils.utils import updateConfig
from models.utils import get_loss_criterion
from models.utils import get_scheduler

from models.meters.moving_avarage_meter import MovingAverageValueMeter
from models.meters.average_meter import AverageValueMeter


class BaseGAN():
    """
    Implementation of DCGAN
    """

    def __init__(self,
                 img_size=64,
                 nz=100,
                 lr=2e-4,
                 beta1=0.5,
                 beta2=0.999,
                 use_gpu=True,
                 loss_criterion="BCE",
                 use_schedulerD=False,
                 use_schedulerG=False,
                 **kwargs):

        self.meterD1 = AverageValueMeter()
        self.meterG1 = AverageValueMeter()

        self.meterD2 = MovingAverageValueMeter(10)
        self.meterG2 = MovingAverageValueMeter(10)

        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.real_label = 1
        self.fake_label = 0

        self.epochs_trained = 0

        if "config" not in vars(self):
            self.config = edict()

        self.config.img_size = img_size
        self.config.nz = nz
        self.config.loss_criterion = loss_criterion

        self.config.lr = float(lr)
        self.config.beta1 = beta1
        self.config.beta2 = beta2

        self.netG = self._get_netG()
        self.netD = self._get_netD()

        self.optimizerG = self._get_optimizerG()
        self.optimizerD = self._get_optimizerD()

        self.loss_criterion = get_loss_criterion(self.config.loss_criterion)

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
             loadConfig=True):

        in_state = torch.load(path)

        self._load_state_dict(in_state,
                              load_netG=load_netG,
                              load_netD=load_netD,
                              load_optimD=load_optimD,
                              load_optimG=load_optimG,
                              loadConfig=loadConfig)

        print(f"Loaded model: {path}")


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
                         loadConfig=True,
                         train=True):

        if loadConfig:
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

        fixed_noise = self.generate_fixed_noise(sample_size)

        with torch.no_grad():
            fake = self.netG(fixed_noise).detach().cpu()

        return fake


    def generate_fixed_noise(self, sample_size=1):
        fixed_noise = torch.randn(sample_size, self.config.nz, 1, 1, device=self.device)
        return fixed_noise


    def _train_step(self, input_batch):

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
        noise = torch.randn(batch_size, self.config.nz, 1, 1, device=self.device)
        # Generate fake image batch with 
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

        # Schedulers updates
        if self.config.use_schedulerD:
            self.schedulerD.step(errD.item())
        if self.config.use_schedulerG:
            self.schedulerG.step(errG.item())

        
        # Apply meters to the losses
        self.meterD1.add(errD)
        self.meterG1.add(errG)

        self.meterD2.add(errD)
        self.meterG2.add(errG)


        # return errD, errG, D_x, D_G_z1, D_G_z2
        return self.meterD1.value(), self.meterG1.value(), D_x, D_G_z1, D_G_z2
        # return self.meterD2.value(), self.meterG2.value(), D_x, D_G_z1, D_G_z2
        







    