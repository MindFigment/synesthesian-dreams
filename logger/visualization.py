import torchvision
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class TensorboardWriter():
    """
    .
    """

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)


    def plot_scalar(self, tag, scalar, epoch):
        self.writer.add_scalar(tag, scalar, epoch)

    
    def plot_losses(self, tag_main, cat1, cat2, scalar1, scalar2, epoch):
        self.writer.add_scalar(tag_main + "/" + cat1, scalar1, epoch)
        self.writer.add_scalar(tag_main + "/" + cat2, scalar2, epoch)

    
    def plot_gans_precisions(self, tag1, tag2, cat2, cat3, scalar1, scalar2, scalar3, epoch):
        self.writer.add_scalar(tag1, scalar1, epoch)
        self.writer.add_scalar(tag2 + "/" + cat2, scalar2, epoch)
        self.writer.add_scalar(tag2 + "/" + cat3, scalar3, epoch)


    def image_summary(self, tag, images):
        # create grid of images
        img_grid = vutils.make_grid(images)

        # write to tensorboard
        self.writer.add_image(tag, img_grid)


    def model_graph(self, model, inputs):
        self.writer.add_graph(model, inputs)


    def images_evolution(self, images):
        pass


    def real_vs_fake(self, tag_main, cat1, cat2, real, fake, epoch):
        # create grid of images
        real_grid = vutils.make_grid(real, normalize=True)

        # create grid of images
        fake_grid = vutils.make_grid(fake, normalize=True)

        # write to tensorboard
        self.writer.add_image(tag_main + "/" + cat1, real_grid, epoch)
        self.writer.add_image(tag_main + "/" + cat2, fake_grid, epoch)
