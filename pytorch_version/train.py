from dataset import ArtDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from models import Discriminator, Generator
from utils import *
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Numbers of workers for dataloader
workers = 2

# Batch size during training
batch_size = 256

# Spatial size of training images. All images will be resized to this size using a transformer
img_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 50

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 100

# Create transformer
transforms_ = [
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

# Create the dataset
# data_root = "../data/images/images"
data_root = "../data/best_artworks_of_all_time/normal"
dataset = ArtDataset(data_root, transforms_=transforms_)

# Define dataset parameters
dataset_params = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": 2
}

# Create the dataloader
dataloader = DataLoader(dataset, **dataset_params)

# Define networks parameters
netD_params = {
    "feature_maps_size": ndf,
    "channels_num": nc
}

netG_params = {
    "latent_vector_size": nz,
    "feature_maps_size": ngf,
    "channels_num": nc
}

# Decide which device you want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Networks
netD = Discriminator(**netD_params).to(device)
netG = Generator(**netG_params).to(device)

netD.apply(weights_init)
netG.apply(weights_init)

# Losses
criterion = nn.BCELoss() 


# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convetion for real and fake labels during training
real_label = 1
fake_label = 0

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Setup Adam optimizers for both G and G
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print(netG)
print(netD)
    
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader):

        ##############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ##############################
        # Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data.to(device)
img_list.        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # calculate gradeints for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ###########################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 1 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' 
                        % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 10 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(fake)

            iters += 1

real_batch = next(iter(dataloader))

plot_loss(G_losses, D_losses)
plot_animation(img_list)
plot_real_vs_fake(real_batch, img_list[-1])
    
