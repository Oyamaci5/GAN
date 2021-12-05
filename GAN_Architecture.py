# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


ngf = 64
ndf = 64
nz = 100

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        self.conv1 = nn.ConvTranspose3d(nz, ngf*16, 4, 1, 0, bias= False )
        
        self.main_conv=nn.Sequential(
            
            nn.ConvTranspose3d(ngf*16, ngf*8, 4, 2, 1, bias = False),
            nn.BatchNorm3d(ngf*8),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(ngf*8, ngf*4, 4, 2, 1, bias = False),
            nn.BatchNorm3d(ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(ngf*4, ngf*2, 4, 2, 1, bias = False),
            nn.BatchNorm3d(ngf*2),
            nn.ReLU(True),
             
            nn.ConvTranspose3d(ngf*2, ngf, 4, 2, 1, bias = False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(ngf, 1, 3, 1, 2, bias=False),
            nn.ReLU()
            
            )
    def forward(self, input):
        return self.main_conv(input)
    

class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        self.main_conv = nn.Sequential(
                nn.Conv3d(1, ndf, 3, 2, 3, bias = False),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv3d(ndf, ndf*2, 3, 2, 1, bias= False),
                nn.BatchNorm3d(ndf*2),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Conv3d(ndf*2, ndf*4, 3, 2, 1, bias= False),
                nn.BatchNorm3d(ndf*4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv3d(ndf*4, ndf*8, 3, 2, 1, bias= False),
                nn.BatchNorm3d(ndf*8),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv3d(ndf*8, 1, 3, 2, 1, bias= False)
                nn.Sigmoid()
                
            )
    def forward(self, input):
        return self.main_conv(input)

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
