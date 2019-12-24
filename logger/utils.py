import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import torchvision.utils as vutils
import numpy as np


def log_train_step_stats(epoch_num, all_epochs, batch_num, all_batches, errD, errG, D_x, D_G_z1, D_G_z2):

    # Output training stats
    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' 
                % (epoch_num, all_epochs, batch_num, all_batches, errD, errG, D_x, D_G_z1, D_G_z2))


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