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
from torch.autograd.variable import Variable
from IPython.display import HTML
import GAN_Architecture as GAN
import sys
from torch.autograd.variable import Variable
ngpu = 1
#Cuda check
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#Workers
workers = 2
#Parameters
batch_size = 32
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
datapath = "Traind_onlyMeV.h5"
with h5py.File(datapath, "r") as f:
    # Get the data
    data = f['Gammaknife/energy']
    dataset = np.array(data)

dataset1 = torch.tensor(dataset)
#Create the dataLoader
dataloader = torch.utils.data.DataLoader(dataset1, batch_size = batch_size, 
                                         shuffle=True, num_workers= workers)
#datset.shape -> torch.Size([199, 90, 90, 90])
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

fixed_noise = torch.randn(batch_size,nz, 1, 1, 1, device = device)
real_label = 1.
fake_label = 0.

#Setup Adam optimizers for both G AND D
optimizerD = optim.Adam(NetD.parameters(), lr=lr, betas=(beta1,0.999))
optimizerG = optim.Adam(NetG.parameters(), lr = lr ,betas=(beta1, 0.999))


#Training
feature_list = []
G_losses = []
D_losses = []


print("Starting Training Loop")
#TODO
#real_cpu = data[0].to(device) - KeyError: 0
iters = 0
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader,0):
        #Train with all-real batch
        NetD.zero_grad()
        # Adversarial ground truths
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        real_cpu = real_cpu.unsqueeze(1)
        valid_label = Variable(FloatTensor(b_size).fill_(1.0), requires_grad=False)
        fake_label = Variable(FloatTensor(b_size).fill_(0.0), requires_grad=False)
       #Train Discriminator
    
        NetD.zero_grad()
      
        output = NetD(real_cpu).view(-1)

       
        errD_real = criterion(output,valid_label)
        #Calculate gradients
        errD_real.backward()
        D_x = output.mean().item()
        
        #Generate batch of latent vectors
        noise = torch.cuda.FloatTensor(b_size, nz, 1, 1, 1, device =device)
        #Generate fake image batch with G
        genlabel = np.random.uniform(10, 100, b_size)
        genlabel = Variable(FloatTensor(genlabel))
        genlabel = genlabel.view(btch_sz, 1, 1, 1, 1)
        fake = NetG(noise.to(device), genlabel)
        #label.fill_(fake_label)
        #Classify all fake batch with D
        output = NetD(fake, genlabel).view(-1)
        #Calculate D's loss on the all fake batch
        errD_fake = criterion(output, fake_label)
        errD_fake.backward(retain_graph=True)
        D_G_z1 = output.mean().item()
        
        #Compute error of D as sum over the fake and real batches
        errD = errD_real + errD_fake
        #Update D
        optimizerD.step()
        
        #Update G
        NetG.zero_grad()
        #label.fill_(fake_label)
        output = NetD(fake, genlabel).view(-1)
        #Calculate G loss
        errG = criterion(output, fake_label)
        #Calculate gradients for G
        errG.backward(retain_graph=True)
        D_G_z2 = output.mean().item()
        #Update G
        optimizerG.step()
        
        #Save losses
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        if (iters % 100 == 0) or ((epoch == num_epochs -1) and i == len(dataloader-1)):
                with torch.no_grad():
                    fake = NetG(fixed_noise, label).detach().cpu()
                feature_list.append(vutils.make_grid(fake, padding = 1, normalize = True))
        
        iters += 1
