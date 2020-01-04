import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from networks.custom_layers import GenGeneralConvBlock, GenInitialBlock, DisGeneralConvBlock, DisFinalBlock

from networks.weights_initializer import weights_init
# from networks.custom_layers import SpectralNorm
from torch.nn.utils import spectral_norm
from torch.nn.utils import remove_spectral_norm


class Generator(nn.Module):
    """ Generator of the MS-GAN network """

    def __init__(self, depth, nz, nc, use_spectral_norm=False):
        super().__init__()

        # print(f"NZ: {nz}")

        assert nz != 0 and ((nz & (nz - 1)) == 0), "latent size not a power of 2"
            
        if depth >= 4:
            assert nz >= np.power(2, depth - 4), "latent size will diminish to zero" # Why?


        self.depth = depth
        self.latent_size = nz
        self.nc = nc
        self.use_spectral_norm = use_spectral_norm

    
        # register the modules required for the Generator Below ...
        # create the ToRGB layers for various outputs
        def to_rgb(in_channels):
            return nn.Conv2d(in_channels, self.nc, (1, 1), bias=True)


        # create a module list of the other required general convolution blocks
        self.layers = nn.ModuleList([GenInitialBlock(self.latent_size)])
        self.rgb_converters = nn.ModuleList([to_rgb(self.latent_size)])

        # create remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size, self.latent_size)
                rgb = to_rgb(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2))
                )
                rgb = to_rgb(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

        if self.use_spectral_norm:
            self.turn_on_spectral_norm()

        self.layers.apply(weights_init)
        self.rgb_converters.apply(weights_init)


    def turn_on_spectral_norm(self):
        for module in self.layers[1:]:
            module.conv_1 = spectral_norm(module.conv_1)
            module.conv_2 = spectral_norm(module.conv_2)
        self.layers[0].tr_conv = spectral_norm( self.layers[0].tr_conv)
        self.layers[0].conv = spectral_norm(self.layers[0].conv)

    
    def turn_off_spectral_norm(self):
        for module in self.layers[1:]:
            module.conv_1 = remove_spectral_norm(module.conv_1)
            module.conv_2 = remove_spectral_norm(module.conv_2)
        self.layers[0].tr_conv = remove_spectral_norm(self.layers[0].tr_conv)
        self.layers[0].conv = remove_spectral_norm(self.layers[0].conv)
            



    def forward(self, x):

        # print(f"INPUT GEN: {x.shape}")
        if self.training:
            outputs = []
            y = x # start computational pipline
            for block, converter in zip(self.layers, self.rgb_converters):
                # print(f"INPUT GEN 1: {y.shape}")
                y = block(y)
                # print(f"INPUT GEN 2: {y.shape}")
                # y = converter(y)
                # print(f"INPUT GEN 2: {y.shape}")
                outputs.append(converter(y))
            return list(reversed(outputs))
        else:
            y = x
            for block in self.layers:
                y = block(y)
            return y
        



class Discriminator(nn.ModuleList):
    """ Discriminator of the MS-GAN network """

    def __init__(self, depth=7, ndf=512, nc=3, use_spectral_norm=False):
        super().__init__()

        assert ndf != 0 and ((ndf & (ndf - 1)) == 0), "latent size not a power of 2"

        # assert depth >= 4, "Depth should be at least 4, to make 64x64 img"

        if depth >= 4:
            assert ndf >= np.power(2, depth - 4), "feature size cannot be produced"


        self.depth = depth
        self.nc = nc
        self.feature_size = ndf
        self.use_spectral_norm = use_spectral_norm


        def from_rgb(out_channels):
            return nn.Conv2d(self.nc, out_channels, (1, 1), bias=True)
        

        # So that first layer feature map size is 16 in case we have depth = 9
        # it's to match generator last layer feature map size 
        # self.initial_feature_size = self.feature_size // np.power(2, self.depth - 4)
        self.initial_converter = from_rgb(self.feature_size // np.power(2, self.depth - 4)) 

        self.layers = nn.ModuleList()
        # Input shape to last block is 515, mid feature size is 512, and output size is 1
        self.final_block = DisFinalBlock(self.feature_size) 

        self.sigmoid = nn.Sigmoid()

        # create remaining layers, in case of depth = 9, go from block 1 to 8
        for i in range(self.depth - 1):
            if i <= 2:
                layer = DisGeneralConvBlock(
                    self.feature_size,
                    self.feature_size,
                    i == self.depth - 2
                ) 
            else:
                layer = DisGeneralConvBlock(
                    self.feature_size // np.power(2, i - 2),
                    self.feature_size // np.power(2, i - 3),
                    i == self.depth - 2
                )       
            self.layers.append(layer)

        if self.use_spectral_norm:
            self.turn_on_spectral_norm()

        self.layers = nn.ModuleList((reversed(self.layers)))

        self.layers.apply(weights_init)

    
    def turn_on_spectral_norm(self):
        for module in self.layers:
            module.conv_1 = spectral_norm(module.conv_1)
            module.conv_2 = spectral_norm(module.conv_2)
        self.final_block.conv_1 = spectral_norm(self.final_block.conv_1)
        self.final_block.conv_2 = spectral_norm(self.final_block.conv_2)

    
    def turn_off_spectral_norm(self):
        for module in self.layers:
            module.conv_1 = remove_spectral_norm(module.conv_1)
            module.conv_2 = remove_spectral_norm(module.conv_2)
        self.final_block.conv_1 = remove_spectral_norm(self.final_block.conv_1)
        self.final_block.conv_2 = remove_spectral_norm(self.final_block.conv_2)


    def forward(self, inputs):

        # print(f"Initial feature size: {self.initial_feature_size}")
        # create a list of downsampled images from the real images:
        # inputs = [x] + [F.avg_pool2d(x, int(np.power(2, i))) for i in range(1, self.depth)]
        # print(f"{len(inputs)} {len(self.layers)}")
        # inputs = list(reversed(inputs))
        # print(f"DISC FORWARD: {len(inputs)}, {inputs[0].shape},  {inputs[-1].shape}, {self.depth}")
        assert len(inputs) == self.depth, "Mismatch between input and Network scales"

        y = self.initial_converter(inputs[0])
        y = self.layers[0](y)

        # print(f"inputs size: {[x.size() for x in inputs]}")
        for input_part, block in zip(inputs[1:-1],
                                    self.layers[1:]):
            # print(f"input part size {input_part.size()}")
            # print(f"OUTPUT SIZE before: {input_part.size()} y = {y.size()}")                          
            y = torch.cat([input_part, y], dim=1)
            # y.register_hook(lambda grad: print(grad))
            # print(f"OUTPUT SIZE after: y = {y.size()}")
            y = block(y)
            # y.register_hook(lambda grad: print(grad))
            # print(f"OUTPUT SIZE block: {y.size()}")

            # print(f"Y SHAPE OUTPUT DISC: {y.shape}")

        # calculate the final block:
        # print("AAAAAAAAAAAA")
        input_part = inputs[-1]
        y = torch.cat([input_part, y], dim=1)
        # print(f"OUTPUT SIZE cat converter: {input_part.size()} {y.size()}")
        y = self.final_block(y)
        # print(f"OUTPUT SIZE final block: {y.size()}")

        # y = self.sigmoid(y)
        # print(f"OUTPUT SIZE sigmoid: {y.size()}")

        # print(f"Y SHAPE OUTPUT DISC: {y.shape}")

        return y


