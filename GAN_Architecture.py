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
        self.conv1 = nn.ConvTranspose3d( nz, ngf * 8, 4, 1, 1, bias=False)
        self.main = nn.Sequential(
            
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
       
            nn.ConvTranspose3d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
          
            nn.ConvTranspose3d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
           
            nn.ConvTranspose3d( ngf, 1, 4, 1, 1, bias=False),
            nn.ReLU()
         
        )
    

    def forward(self, noise, energy):
        input = self.conv1(noise * energy)
        return self.main(input)

    

class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        self.main = nn.Sequential(
                nn.Conv3d(1, ndf, 3 , 2, 1, bias = False),
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
                
                nn.Conv3d(ndf*8, ndf*4, 3, 2, 1, bias = False),
                nn.Sigmoid()
              
                
        )
       
              
    def forward(self, energy):
        return self.main(energy)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.2)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    
