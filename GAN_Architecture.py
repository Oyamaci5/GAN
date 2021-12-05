# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


ngf = 32
ndf = 32
nz = 100

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose3d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose3d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose3d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        self.main = nn.Sequential(
                nn.Conv3d(3, ndf, 4, 2, 1, bias = False),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv3d(ndf, ndf*2, 3, 2, 1, bias= False),
                nn.BatchNorm3d(ndf*2),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Conv3d(ndf*2, ndf*4, 3, 2, 1, bias= False),
                nn.BatchNorm3d(ndf*4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv3d(ndf*4, ndf*8, 3, 2, 1, bias= False),
                nn.BatchNorm3d(ndf *8 ),
                nn.LeakyReLU(0.2, inplace = True),
                
                nn.Conv3d(ndf*8, 1, 3, 1, 0, bias = False),
                nn.Sigmoid()
              
                
            )
       
                
    def forward(self, input):
         return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.2)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear')!= -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
