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
import click

@click.command()
@click.option("--epochs", default=100, help="Number of epochs to train the model")
@click.option("--batch_size", default=256, help="Batch size")
@click.option("--nz", default=100, help="Size of the latent vector")
@click.option("--use_gpu/--use_cpu", is_flag=True, default=False)
@click.option("--model_dir", default="toy_model")
@click.option("--model_name", default="toy_model")
@click.option("--load_model", is_flag=True, default=False, help="Set if you want to load existing model called model_name from model_dir")
@click.option("--save_model", is_flag=True, default=False, help="Set if you want to save model during training")
def train(epochs, batch_size, nz, use_gpu, model_dir, model_name, load_model, save_model):

    # Numbers of workers for dataloader
    workers = 2

    # batch_size: batch size during training

    # Spatial size of training images. All images will be resized to this size using a transformer
    img_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # nz: size of z latent vector (i.e. size of generator input)

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = epochs

    # Create transformer
    transforms_ = [
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # Create the dataset
    if use_gpu:
        data_root = "./data/best_artworks_of_all_time/normal"
    else:
        data_root = "./data/images/images"

    dataset = ArtDataset(data_root, transforms_=transforms_)

    # Define dataset parameters
    dataset_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 2
    }

    # Create the dataloader
    dataloader = DataLoader(dataset, **dataset_params)


    model_dir = os.path.join("./models", model_dir)
    os.makedirs(model_dir, exist_ok=True) 

    model_path = os.path.join(model_dir, model_name)

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

    # Learning rate for optimizers
    lr = 0.0002
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # Setup Adam optimizers for both G and G
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    start = None
    end = None

    if load_model:
        
        checkpoint = torch.load("".join([model_path, ".pt"]))

        print(f"Loading model {model_path}")

        netD.load_state_dict(checkpoint["netD_state_dict"])
        netG.load_state_dict(checkpoint["netG_state_dict"])

        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])

        # netD.eval()
        # netG.eval()
        # - or -
        netD.train()
        netG.train()

        start = checkpoint["epoch"]
        end = num_epochs + start

    else:
        netD.apply(weights_init)
        netG.apply(weights_init)

        start = 0
        end = num_epochs

    # Losses
    criterion = nn.BCELoss() 


    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convetion for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print(netG)
    print(netD)

    print(f"Data root: {data_root}, use gpu: {use_gpu}")
    print(f"Model path: {model_path}")
    print(f"Save model {save_model}")
        
    # For each epoch
    for epoch in range(start, end):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader):

            ##############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ##############################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
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
                            % (epoch, end, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 300 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(fake)

                if save_model:
                    model_checkpoint = "".join([model_path, "_", str(epoch), ".pt"])
                    print(f"Saving model... to {model_checkpoint}")
                    torch.save({
                        "netD_state_dict": netD.state_dict(),
                        "netG_state_dict": netG.state_dict(),
                        "optimizerD_state_dict": optimizerD.state_dict(),
                        "optimizerG_state_dict": optimizerG.state_dict(),
                        "epoch": epoch,
                        "lossD": errD.item(),
                        "lossG": errG.item()
                        }, model_checkpoint)

            iters += 1

    real_batch = next(iter(dataloader))

    plot_loss(G_losses, D_losses, plot_dir=model_name)
    plot_animation(img_list[-16:], plot_dir=model_name)
    plot_real_vs_fake(real_batch, img_list[-1], plot_dir=model_name)


if __name__ == "__main__":
    train()

    
