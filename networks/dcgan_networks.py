import torch.nn as nn
from torchsummary import summary
import numpy as np

from networks.utils import feature_maps_multiplies, multiplies
from networks.weights_initializer import weights_init



class Discriminator(nn.Module):
    def __init__(self,
                 img_size,
                 ndf, 
                 nc):
        super(Discriminator, self).__init__()
        
        self.img_size = img_size
        self.ndf = ndf
        self.nc = nc 
        self.main = nn.Sequential(
            self._make_layers(self.img_size, self.ndf, self.nc))
        
        summary(self.main.cuda(), (self.nc, self.img_size, self.img_size))

        self.main.apply(weights_init)


    def forward(self, input):
        return self.main(input)
    
    
    def _make_layers(self, img_size, ndf, nc):
    
        layers = []

        layers += [ nn.Conv2d(nc, ndf, 4, 2, 1, bias=False) ]
        layers += [ nn.LeakyReLU(0.2, inplace=True) ]

        multiplies = feature_maps_multiplies(img_size)

        for m in multiplies:
            layers += [ nn.Conv2d(ndf * m // 2, ndf * m, 4, 2, 1, bias=False) ]
            layers += [ nn.BatchNorm2d(ndf * m) ]
            layers += [ nn.LeakyReLU(0.2, inplace=True) ]

        layers += [ nn.Conv2d(ndf * multiplies[-1], 1, 4, 1, 0, bias=False) ]
        layers += [ nn.Sigmoid() ]
    
        return nn.Sequential(*layers)



class Generator(nn.Module):
    def __init__(self,
                 img_size,
                 nz,
                 ngf, 
                 channels_num):
        super(Generator, self).__init__()
        
        self.img_size = img_size
        self.nz = nz
        self.ngf = ngf
        self.nc = channels_num

        self.main = self._make_layers(self.img_size, self.nz, self.ngf, self.nc)
        
        summary(self.main.cuda(), (self.nz, 1, 1))
        
        self.main.apply(weights_init)


    def forward(self, input):
        return self.main(input)
    
    
    def _make_layers(self, img_size, nz, ngf, nc):
    
        multiplies = list(reversed(feature_maps_multiplies(img_size)))
    
        layers = []

        layers += [ nn.ConvTranspose2d(nz, ngf * multiplies[0], 4, 1, 0, bias=False) ]
        layers += [ nn.BatchNorm2d(ngf * multiplies[0])]
        layers += [ nn.ReLU(True) ]

        for m in multiplies:
            layers += [ nn.ConvTranspose2d(ngf * m, ngf * m // 2, 4, 2, 1, bias=False) ]
            layers += [ nn.BatchNorm2d(ngf * m // 2) ]
            layers += [ nn.ReLU(True) ]

        layers += [ nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False) ]
        layers += [ nn.Tanh() ]
    
        return nn.Sequential(*layers)