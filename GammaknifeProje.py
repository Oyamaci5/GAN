from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import HDF5Dataset as H
import GAN_Architecture as GAN
import sys
ngpu = 1
#Cuda check
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#Workers
workers = 0
#Parameters
batch_size = 256
feature_size = 90
nz = 100
#Number of channels 
nc = 4
#Epochs
num_epochs = 10
#Learning Rate
lr= 0.0002
#Beta1 hyperparam for Adam optimizers
beta1 = 0.5
#PATHS
PATH_save = 'Save_model/conv3d_corr_new2.pth'
PATH_chechpoint = 'Save_model/new2_1.pth'
#Dataset
#Gammaknife is the group name
datapath = "Traindata.h5"
dataset = H.HDF5Dataset(datapath, 'Gammaknife')
#Create the dataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                         shuffle=True, num_workers= workers)
#Decide which device (GPU)
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu>0) else "cpu")


#Create the generator
NetG = GAN.Generator(ngpu).to(device)
if(device.type == 'cuda') and (ngpu > 1):
    NetG = nn.DataParallel(NetG, list(range(ngpu)))


NetG.apply(GAN.weights_init)
#Print the Generator model
print(NetG)
NetD = GAN.Discriminator(ngpu).to(device)
if(device.type =='cuda') and (ngpu > 1):
    NetD = nn.DataParallel(NetD, list(range(ngpu)))

NetD.apply(GAN.weights_init)
#Print the Discriminator model
print(NetD)

#Initialize BCELoss(Binary Cross Entropy Loss) function
criterion = nn.BCELoss()

fixed_noise = torch.randn(128,nz, 1, 1, device = device)
real_label = 1.
fake_label = 0.

#Setup Adam optimizers for both G AND D
optimizerD = optim.Adam(NetD.parameters(), lr=lr, betas=(beta1,0.999))
optimizerG = optim.Adam(NetG.parameters(), lr = lr ,betas=(beta1, 0.999))


#Training
feature_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop")
#TODO
#real_cpu = data[0].to(device) - KeyError: 0
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        #Train with all-real batch
        NetD.zero_grad()
        
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype= torch.float,
                           device = device)
        
        output = NetD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        #Calculate gradients
        errD_real.backward()
        D_x = output.mean().item()
        
        #Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device =device)
        #Generate fake image batch with G
        fake = NetG(noise)
        label.fill_(fake_label)
        #Classify all fake batch with D
        output = NetD(fake.detach()).view(-1)
        #Calculate D's loss on the all fake batch
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        #Compute error of D as sum over the fake and real batches
        errD = errD_real + errD_fake
        #Update D
        optimizerD.step()
        
        #Update G
        NetG.zero_grad()
        label.fill_(real_label)
        output = NetD(fake).view(-1)
        #Calculate G loss
        errG = criterion(output, label)
        #Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        #Update G
        optimizerG.step()
        
        #Save losses
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        if (iters % 100 == 0) or ((epoch == num_epochs -1) and i == len(dataloader-1)):
                        with torch.no_grad():
                            fake = NetG(fixed_noise).detach().cpu()
                        feature_list.append(vutils.make_grid(fake, padding = 2, normalize = True))
        
        iters += 1
