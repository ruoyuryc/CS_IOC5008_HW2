#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import datetime
import os, sys
import glob
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, imsave

import math
import hashlib
from urllib.request import urlretrieve
import zipfile
import gzip
import shutil
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


MODEL_NAME = 'BEGAN'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


IMAGE_DIM = (64, 64, 3)


# In[5]:


def tensor2img(tensor):
    img = (np.transpose(tensor.detach().cpu().numpy(), [1,2,0])+1)/2.
    return img


# In[6]:


def get_sample_image(G, n_noise=100, n_samples=64):
    """
        save sample 100 images
    """
    n_rows = int(np.sqrt(n_samples))
    z = (torch.rand(size=[n_samples, n_noise])*2-1).to(DEVICE) # U[-1, 1]
    x_fake = G(z)
    x_fake = torch.cat([torch.cat([x_fake[n_rows*j+i] for i in range(n_rows)], dim=1) for j in range(n_rows)], dim=2)
    result = tensor2img(x_fake)
    return result


# In[7]:


class Encoder(nn.Module):
    def __init__(self, in_channel=1, n_filters=128, hidden_dim=100):
        super(Encoder, self).__init__()
        assert IMAGE_DIM[0] % 2**2 == 0, 'Should be divided 4'
        self.flatten_dim = IMAGE_DIM[0]//2**2
        self.conv = nn.Sequential(
            # conv0
            nn.Conv2d(in_channel, n_filters, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),
            # conv1
            nn.Conv2d(n_filters, n_filters, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),
            # conv2
            nn.Conv2d(n_filters, n_filters*2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_filters*2),
            nn.LeakyReLU(0.2),
            # conv3
            nn.Conv2d(n_filters*2, n_filters*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters*2),
            nn.LeakyReLU(0.2),
            # conv4
            nn.Conv2d(n_filters*2, n_filters*3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_filters*3),
            nn.LeakyReLU(0.2),
            # conv5
            nn.Conv2d(n_filters*3, n_filters*3, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters*3),
            nn.LeakyReLU(0.2),
            # conv6
            nn.Conv2d(n_filters*3, n_filters*3, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters*3),
            nn.LeakyReLU(0.2),
#             nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(self.flatten_dim**2*n_filters*3, hidden_dim)
    
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h


# In[8]:


class Decoder(nn.Module):
    """
        Convolutional Decoder
    """
    def __init__(self, out_channel=1, n_filters=128, n_noise=100):
        super(Decoder, self).__init__()
        assert IMAGE_DIM[0] % 2**2 == 0, 'Should be divided 4'
        self.flatten_dim = IMAGE_DIM[0]//2**2
        self.fc = nn.Sequential(
            nn.Linear(n_noise, self.flatten_dim**2*n_filters),
        )
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(n_filters, n_filters, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),
            # conv2
            nn.Conv2d(n_filters, n_filters, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # conv3
            nn.Conv2d(n_filters, n_filters, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),
            # conv4
            nn.Conv2d(n_filters, n_filters, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # conv5
            nn.Conv2d(n_filters, n_filters, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),
            # conv6
            nn.Conv2d(n_filters, n_filters, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),
            # conv6
            nn.Conv2d(n_filters, out_channel, 3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )
    
    def forward(self, h):
        h = self.fc(h)
        h = h.view(h.size(0), -1, self.flatten_dim, self.flatten_dim)
        x = self.conv(h)
        return x


# In[9]:


class Discriminator(nn.Module):
    """
        Convolutional Discriminator
    """
    def __init__(self, in_channel=1, n_filters=128, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(in_channel=in_channel, n_filters=n_filters, hidden_dim=hidden_dim)
        self.decoder = Decoder(out_channel=in_channel, n_filters=n_filters, n_noise=hidden_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        x_ = self.decoder(h)
        return x_


# In[10]:


class Generator(nn.Module):
    """
        Convolutional Generator
    """
    def __init__(self, out_channel=1, n_filters=128, n_noise=64):
        super(Generator, self).__init__()
        self.decoder = Decoder(out_channel=out_channel, n_filters=n_filters, n_noise=n_noise)
        
    def forward(self, h):
        x_ = self.decoder(h)
        return x_


# In[11]:


transform = transforms.Compose([transforms.Resize((IMAGE_DIM[0],IMAGE_DIM[1])),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5))
                               ]
)


# In[12]:


dataset = datasets.ImageFolder(root='/home/haaaaa/hw2/new_data_3', transform=transform)


# In[13]:


batch_size = 32
n_noise = 64


# In[14]:


data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)


# In[15]:


D = Discriminator(in_channel=IMAGE_DIM[-1], hidden_dim=n_noise).to(DEVICE)
G = Generator(out_channel=IMAGE_DIM[-1], n_noise=n_noise).to(DEVICE)


# In[16]:


D_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
G_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))


# In[17]:


# BEGAN causes mode collapse. it can be addressed by decaying lr
D_scheduler = torch.optim.lr_scheduler.MultiStepLR(D_opt, milestones=[3, 10, 17, 24, 32], gamma=0.6)
G_scheduler = torch.optim.lr_scheduler.MultiStepLR(G_opt, milestones=[3, 10, 17, 24, 32], gamma=0.6)


# In[18]:


criterion = nn.L1Loss()


# In[29]:


max_epoch = 7
step = 0


# In[30]:


lr_k = 0.001
gamma = 0.7
k_t = 0
log_term = 1000


# In[31]:


steps_per_epoch = len(data_loader.dataset) // batch_size
steps_per_epoch


# In[32]:


if not os.path.exists('samples'):
    os.makedirs('samples')
    
if not os.path.exists('ckpt'):
    os.makedirs('ckpt')


# In[33]:


def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    torch.save(state, file_name)


# In[34]:


m_lst = []
bestM = 1.
for epoch in range(max_epoch):
    for idx, (images, labels) in enumerate(data_loader):
        G.zero_grad()
        # Training Discriminator
        x = images.to(DEVICE)
        x_outputs = D(x)
        D_x_loss = criterion(x_outputs, x)

        z = (torch.rand(size=[batch_size, n_noise])*2-1).to(DEVICE)
        x_fake = G(z)
        z_outputs = D(x_fake.detach())
        D_z_loss = criterion(z_outputs, x_fake)
        
        D_loss = D_x_loss - k_t*D_z_loss
        
        D.zero_grad()
        D_loss.backward()
        D_opt.step()

        # Training Generator
        z = (torch.rand(size=[batch_size, n_noise])*2-1).to(DEVICE)
        x_fake = G(z)
        z_outputs = D(x_fake)
        G_loss = criterion(x_fake, z_outputs)

        G.zero_grad()
        G_loss.backward()
        G_opt.step()
        
        bal = (gamma*D_x_loss - G_loss).detach()
        k_t = min(max(k_t + lr_k*bal, 0), 1)
        M_global = D_x_loss.detach() + torch.abs(bal)
        
        if M_global.item() < bestM:
            bestM = M_global.item()
            save_checkpoint({'global_step': step,
                 'D':D.state_dict(),
                 'G':G.state_dict(),
                 'd_optim': D_opt.state_dict(),
                 'g_optim' : G_opt.state_dict()},
                'ckpt/began{:06d}.pth.tar'.format(step))
        
        if step % log_term == 0:
            m_lst.append(M_global)
            dt = datetime.datetime.now().strftime('%H:%M:%S')
            print('Epoch: {}/{}, Step: {}, D Loss: {:.4f}, G Loss: {:.4f}, k: {:.4f}, M: {:.4f}, lr: {:.7f}, Time:{}'.format(epoch, max_epoch, step, D_loss.item(), G_loss.item(), k_t, M_global.item(), G_scheduler.get_lr()[0], dt))
            G.eval()
            img = get_sample_image(G, n_noise, n_samples=25)
            imsave('samples/{}_step{:06d}.jpg'.format(MODEL_NAME, step), img)
            G.train()
        
        step += 1
    D_scheduler.step()
    G_scheduler.step()


# In[35]:


def get_batch(image_files, width, height, mode):
    data_batch = np.array(
        [get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)

    # Make sure the images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch


def images_square_grid(images):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    # Scale to 0-255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # Put images in a square arrangement
    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))

    # Combine images to grid image
    new_im = Image.new('RGB', (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, 'RGB')
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im


# In[36]:


img = get_sample_image(G, n_noise, n_samples=9)
imshow(img)


# In[37]:


def output_fig(images_array, file_name="./results123"):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(images_square_grid(a))
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)


# In[38]:


import matplotlib
for r in range(5):
    img = []
    for l in range(9):
        i = get_sample_image(G, n_noise, n_samples=1)
        img.append(i)
    a = np.array(img)
    output_fig(a,'./'+str(r))


# In[ ]:




