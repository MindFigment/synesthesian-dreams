import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.utils as vutils
from torchvision import transforms, utils
from dataset import ArtDataset
from torch.utils.data import Dataset, DataLoader
import os 


# Custom weights initialization called netG and netG
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_loss(G_losses, D_losses, plot_dir="toy_model", plot_name="losses_plot.png", show=False):

    plot_path = os.path.join("./images", plot_dir)
    os.makedirs(plot_path, exist_ok=True) 
    save_as = os.path.join(plot_path, plot_name)

    _ = plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    if show:
        plt.show()
    plt.savefig(save_as)


def plot_animation(img_list, plot_dir="toy_model", plot_name="real_vs_fake.gif", show=False):

    plot_path = os.path.join("./images", plot_dir)
    os.makedirs(plot_path, exist_ok=True) 
    save_as = os.path.join(plot_path, plot_name)

    fig = plt.figure(figsize=(20,20))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(vutils.make_grid(i, padding=2, nrow=int(np.sqrt(i.size(0))), normalize=True), (1,2,0)), animated=True)] for i in img_list]
    anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    if show:
        plt.show()
    anim.save(save_as, writer='imagemagick', dpi=30)


def plot_real_vs_fake(real_imgs, fake_imgs, plot_dir="toy_model", plot_name="fake_vs_real.png", show=False):
    
    plot_path = os.path.join("./images", plot_dir)
    os.makedirs(plot_path, exist_ok=True) 
    save_as = os.path.join(plot_path, plot_name)

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_imgs[:16], nrow=int(np.sqrt(real_imgs[:16].size(0))), padding=5, normalize=True), (1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(fake_imgs[:16], nrow=int(np.sqrt(fake_imgs[:16].size(0))), padding=5, normalize=True),(1,2,0)))
    if show:
        plt.show()
    plt.savefig(save_as)


if __name__ == "__main__":

    img_size = 64
    batch_size = 32

    # Create transformer
    transforms_ = [
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # Create the dataset
    data_root = "./data/images/images"
    # data_root = "../data/best_artworks_of_all_time/normal"
    dataset = ArtDataset(data_root, transforms_=transforms_)

    # Define dataset parameters
    dataset_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 2
    }

    # Create the dataloader
    dataloader = DataLoader(dataset, **dataset_params)


    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    print(real_batch.size())
    # real_batch[0].to(device)[:64]
