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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animatio
from IPython.display import HTML
import sys
import h5py
import seaborn as sns
import torch.utils.data as Dataset
from torch.autograd.variable import Variable
ngpu = 1
#Cuda check
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#Workers
workers = 2
#Parameters
batch_size = 8
feature_size = 90
nz = 100
ngf = 8
ndf = 8
#Number of channels 
nc = 4
#Epochs
num_epochs = 10
#Learning Rate
lr= 0.0002
#Beta1 hyperparam for Adam optimizers
beta1 = 0.5
#PATHS
PATH_save = '/content/gdrive/MyDrive/Model/conv3d_corr_new2.pth'
PATH_chechpoint = '/content/gdrive/MyDrive/Model/new2_1.pth'
torch.cuda.empty_cache()
#save function
def save(netG, netD, optim_G, optim_D, epoch, loss, path_to_save):
    torch.save({
                'Generator': NetG.state_dict(),
                'Discriminator': NetD.state_dict(),
                'G_optimizer': optim_G.state_dict(),
                'D_optimizer': optim_D.state_dict(),
                #'D_scores': scores,
                'epoch': epoch,
                'loss': loss,
                },
                path_to_save)
#GAN architecture
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.ConvTranspose3d( nz, ngf * 8, 4, 1, 0, bias=False)
        self.main = nn.Sequential(

            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
       
            nn.ConvTranspose3d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
          
            nn.ConvTranspose3d( ngf * 2, ngf, 4, 3, 2, bias=False),
            nn.BatchNorm3d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
           
            nn.ConvTranspose3d( ngf, nc, 4, 2, 1, bias=False),
            #nn.Tanh(),
            nn.ReLU(),
        )
    

    def forward(self,energy):
        input = self.conv1(energy)
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        self.main = nn.Sequential(
                nn.Conv3d(1, ndf, 4 , 2, 1, bias = False),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv3d(ndf, ndf*2, 4, 2, 1, bias= False),
                nn.BatchNorm3d(ndf*2),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Conv3d(ndf*2, ndf*4, 4, 2, 1, bias= False),
                nn.BatchNorm3d(ndf*4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv3d(ndf*4, ndf*8, 4, 2, 1, bias= False),
                nn.BatchNorm3d(ndf *8 ),
                nn.LeakyReLU(0.2, inplace = True),
                
                nn.Conv3d(ndf*8, 1, 4, 2, 0, bias = False),
                nn.Sigmoid()
                )
    def forward(self, energy):
        return self.main(energy)
       # return self.embed(conv_out).view(-1,ndf*4)
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
#Dataset
#HDF5 function

class HDF5Dataset(Dataset):
    def __init__(self, path):
        self.file_path = path
        self.dataset = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["energy"])

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')

        return {'energy' : self.dataset['energy'][index]}

    def __len__(self):
        return self.dataset_len
#Gammaknife is the group name
path = '/content/gdrive/MyDrive/Datas/Train3d_onlyentry.h5'#Train2d_onlyentry
data = HDF5Dataset(path)
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
#datset.shape -> torch.Size([199, 90, 90, 90])
#Decide which device (GPU)
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu>0) else "cpu")


#Create the generator
#Create the generator
NetG = Generator(ngpu).to(device)
if(device.type == 'cuda') and (ngpu > 1):
    NetG = nn.DataParallel(NetG, list(range(ngpu)))


NetG.apply(weights_init)
#Print the Generator model
print(NetG)
NetD = Discriminator(ngpu).to(device)
if(device.type =='cuda') and (ngpu > 1):
    NetD = nn.DataParallel(NetD, list(range(ngpu)))

NetD.apply(weights_init)
#Print the Discriminator model
print(NetD)

#Initialize BCELoss(Binary Cross Entropy Loss) function
criterion = nn.BCELoss()

fixed_noise = torch.randn(8,nz, 1, 1, 1, device = device)

#Setup Adam optimizers for both G AND D
optimizerD = optim.Adam(NetD.parameters(), lr
                        =lr, betas=(beta1,0.999))
optimizerG = optim.Adam(NetG.parameters(), lr = lr ,betas=(beta1, 0.999))
#Training
NetG.train()
NetD.train()
#Training
feature_list = []
G_losses = []
D_losses = []
print("Starting Training Loop")


print("Starting Training Loop")
#TODO
#real_cpu = data[0].to(device) - KeyError: 0
iters = 0
iters = 0
img_list = []
for epoch in range(num_epochs+75):
    for i, batch in enumerate(dataloader,0):
        btch_sz = len(batch['energy'])
        real_showers = Variable(batch['energy']).float().to(device).view(btch_sz, 1, 90, 90, 90)

        valid_label = Variable(FloatTensor(btch_sz).fill_(0.8), requires_grad=False)
        fake_label = Variable(FloatTensor(btch_sz).fill_(0.2), requires_grad=False)
        ######################################################
        # Train Discriminator
        ######################################################
        NetD.zero_grad()

        output = NetD(real_showers).view(-1)
        errD_real = criterion(output, valid_label)
        errD_real.backward()
        D_x = output.mean().item()
        noise = torch.FloatTensor(btch_sz, 100, 1, 1, 1).uniform_(-1, 1)
        #Generate fake image
        fake = NetG(noise.to(device))
        output = NetD(fake.detach()).view(-1)
        errD_fake = criterion(output, fake_label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        NetG.zero_grad()
        output = NetD(fake).view(-1)
        errG = criterion(output, valid_label)
        errG.backward(retain_graph=True)
        D_G_z2 = output.mean().item()
        optimizerG.step()
        if i % 10 == 0:
          print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader),
                  errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
          with torch.no_grad():
            img_list.append(fake)
        #Save losses
        G_losses.append(errG.item())
        D_losses.append(errD.item())
    loss =  np.array([G_losses, D_losses])
    save(netG=NetG, netD=NetD, optim_G=optimizerG, optim_D=optimizerD, epoch=epoch, loss=loss,
    path_to_save=PATH_save)
    
#plotting lost of Discriminator and Genarator
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show() 

#To show save data
count = 0
for y in img_list:
  y = y.squeeze(1)
  y = torch.Tensor.detach(y).cpu().numpy()
    
data = y[0]

df = pd.DataFrame(columns = ['X', 'Y','Z', 'Entry'])
df.astype({"X": int, "Y": int,'Z': int, 'Entry':float})
for i in range(0, 90, 1):
  for j in range(0,90,1):
    for s in range(0,90,1):
      df = df.append({'X' : i, 'Y' : j,'Z':s, 'Entry' : data[i][j][s]}, 
                ignore_index = True)

sns.pairplot(df[['X','Y','Entry']], diag_kind='kde')
