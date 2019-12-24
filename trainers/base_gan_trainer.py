from models.dcgan import DCGAN
from logger.utils import log_train_step_stats
from datasets.art_dataset import ArtDataset
from logger.visualization import TensorboardWriter
import numpy as np
import os
from datetime import datetime

from easydict import EasyDict as edict

import torch
from torchvision import transforms, utils

class BaseGANTrainer():
    """
    .
    """

    def __init__(self,
                 t_c,
                 m_c
                 ):

        if "t_c" not in vars(self):
            self.t_c = edict()

            self.t_c.want_log = t_c.want_log

            self.t_c.img_size = t_c.img_size
            
            self.t_c.save = t_c.save
            self.t_c.save_every = t_c.save_every
            self.t_c.save_ext = t_c.save_ext

            self.t_c.load = t_c.load
            if self.t_c.load:
                self.t_c.load_model = t_c.load_model
                self.t_c.load_netD = t_c.load_netD
                self.t_c.load_netG = t_c.load_netG
                self.t_c.load_optimD = t_c.load_optimD
                self.t_c.load_optimG = t_c.load_optimG

            self.t_c.test = t_c.test
            self.t_c.test_every = t_c.test_every
            self.t_c.sample_size = t_c.sample_size

            self.t_c.batch_size = t_c.batch_size
            self.t_c.shuffle = t_c.shuffle
            self.t_c.num_workers = t_c.num_workers
            self.t_c.epochs = t_c.epochs

            self.t_c.summary_dir = t_c.summary_dir
            self.t_c.checkpoint_dir = t_c.checkpoint_dir
            self.t_c.log_dir = t_c.log_dir
            self.t_c.out_dir = t_c.out_dir

            self.t_c.data_roots = t_c.data_roots

        if "m_c" not in vars(self):
            self.m_c = edict(m_c)

        self.summary_writer = TensorboardWriter(self.t_c.summary_dir)
        
        self.init_model()


    def init_model(self):
        raise NotImplementedError

    
    def run(self):
        """
        The main operator
        """
        try:
            self.train()
        except KeyboardInterrupt:
            print("You gave entered CTRL+C... Wait to finalize")


    def train(self):
        raise NotImplementedError


    def _train_one_epoch(self, dataloader, epoch):

        all_batches = len(dataloader)

        # For each batch in the dataloader
        for batch_num, batch_data in enumerate(dataloader):
            
            errD, errG, D_x, D_G_z1, D_G_z2 = self.model._train_step(batch_data)
            # Log batch stats into terminal
            log_train_step_stats(epoch, self.end, batch_num, all_batches, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)

            if self.t_c.want_log:
                self.summary_writer.plot_losses("D loss vs G loss", "D", "G", errD.item(), errG.item(), epoch)
                self.summary_writer.plot_gans_precisions("D_x", "D_G_z", "Before train step", "After train step", D_x, D_G_z1, D_G_z2, epoch)
                self.summary_writer.plot_scalar("After minus before step", D_G_z2 - D_G_z1, epoch)

        # Save model
        if self.t_c.save and epoch % self.t_c.save_every == 0:
            self.model.save(self.t_c.checkpoint_dir, self.t_c.save_ext)

        # Test model
        if self.t_c.test and epoch % self.t_c.test_every == 0:
            fake_sample = self.model.generate_images(sample_size=self.t_c.sample_size)
            r_i = np.random.choice(len(batch_data), len(fake_sample))
            real_sample = batch_data[r_i]

            if self.t_c.want_log:
                self.summary_writer.real_vs_fake(f"Real vs fake", "real", "fake", real_sample, fake_sample, epoch)
            # self.tensorboard.model_graph(self.model.netD, batch_data.to(self.model.device))
            # self.tensorboard.model_graph(self.model.netG, self.model.generate_fixed_noise(len(batch_data)))
        

        self.model.epochs_trained += 1


    def _get_dataloader(self):
        
        dataset = self._get_dataset()
        dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=self.t_c.batch_size,
                                            shuffle=self.t_c.shuffle,
                                            num_workers=self.t_c.num_workers)
        
        return dataloader


    def _get_dataset(self):
        dataset = ArtDataset(self.t_c.data_roots, transforms_= [
                                            transforms.Resize(self.t_c.img_size),
                                            transforms.CenterCrop(self.t_c.img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

        return dataset