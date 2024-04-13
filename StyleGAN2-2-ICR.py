#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from comet_ml import Experiment
#experiment = Experiment()


# In[ ]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from torch import einsum
from einops import rearrange
from tqdm import tqdm
from PIL import Image, ImageFile
from pickle import load, dump
import cv2
import random
import time
import argparse
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


class WeightScaledConv2d(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_nc, input_nc//groups, kernel_size, kernel_size))
        self.scale = np.sqrt(2 / (input_nc * kernel_size ** 2))
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.upsample = False
        
    def deconv(self):
        self.upsample = True
        return self
    
    def forward(self, x):
        weight = self.weight * self.scale
        if not self.upsample:
            out = F.conv2d(x, weight=weight, stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            weight = weight.transpose(0, 1)
            out = F.conv_transpose2d(x, weight=weight, stride=self.stride, padding=self.padding, groups=self.groups)
        return out


# In[ ]:


class WeightScaledLinear(nn.Module):
    def __init__(self, input_nc, output_nc, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_nc, input_nc))
        self.scale = np.sqrt(2 / input_nc)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_nc))
        else:
            self.bias = None
    
    def forward(self, x):
        weight = self.weight * self.scale
        out = F.linear(x, weight=weight, bias=self.bias)
        return out


# In[ ]:


class GLU(nn.Module):
    def forward(self, x):
        channel = x.size(1)
        assert channel % 2 == 0, 'must divide by 2.'
        return x[:, :channel//2] * torch.sigmoid(x[:, channel//2:])


# In[ ]:


class FReLU(nn.Module):
    def __init__(self, n_channel, kernel=3, stride=1, padding=1):
        super().__init__()
        self.funnel_condition = WeightScaledConv2d(n_channel, n_channel, kernel_size=kernel,stride=stride, padding=padding, groups=n_channel)
        self.bn = nn.BatchNorm2d(n_channel)

    def forward(self, x):
        tx = self.bn(self.funnel_condition(x))
        out = torch.max(x, tx)
        return out


# In[ ]:


class Mish(nn.Module):
    @staticmethod
    def mish(x):
        return x * torch.tanh(F.softplus(x))
    
    def forward(self, x):
        return Mish.mish(x)


# In[ ]:


class SelfAttention(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        
        # Pointwise Convolution
        self.query_conv = nn.Conv2d(input_nc, input_nc // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(input_nc, input_nc // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(input_nc, input_nc, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, return_map=False):
        proj_query = self.query_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3]).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3])
        s = torch.bmm(proj_query, proj_key)
        attention_map_T = self.softmax(s)
        
        proj_value = self.value_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3])
        o = torch.bmm(proj_value, attention_map_T)
        
        o = o.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        out = x + self.gamma * o
        
        if return_map:
            return out, attention_map_T.permute(0, 2, 1)
        else:
            return out


# In[ ]:


class PixelwiseNormalization(nn.Module):
    def pixel_norm(self, x):
        eps = 1e-8
        return x * torch.rsqrt(torch.mean(x * x, 1, keepdim=True) + eps)
    
    def forward(self, x):
        return self.pixel_norm(x)


# In[ ]:


class ModulatedConv2d(nn.Module):
    def __init__(self, input_nc, output_nc, dim_latent, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.epsilon = 1e-8
        self.stride = stride
        self.padding = padding
        
        self.norm = nn.InstanceNorm2d(input_nc)
        self.weight = nn.Parameter(torch.randn([1, output_nc, input_nc, kernel_size, kernel_size]))
        self.scale = np.sqrt(2 / (input_nc * kernel_size ** 2))
        self.modulate = WeightScaledLinear(dim_latent, input_nc)
        
        self.upsample = False
        
    def deconv(self):
        self.upsample = True
        return self
        
    def forward(self, image, style):
        batch, input_nc, height, width = image.shape
        _, output_nc, _, kernel_size, _ = self.weight.shape
        
        style = self.modulate(style)
        weight = self.weight * self.scale * style.view(batch, 1, input_nc, 1, 1)
        
        # demodulation
        demodulate = (self.weight.square().sum([2, 3, 4]) + self.epsilon).rsqrt().view(1, output_nc, 1, 1, 1)
        weight = weight * demodulate
        
        image = image.reshape(1, batch * input_nc, height, width)
        
        if not self.upsample:
            weight = weight.view(batch * output_nc, input_nc, kernel_size, kernel_size)
            out = F.conv2d(image, weight=weight, bias=None, groups=batch, stride=self.stride, padding=self.padding)
        else:
            weight = weight.transpose(1, 2)
            weight = weight.reshape(batch * input_nc, output_nc, kernel_size, kernel_size)
            out = F.conv_transpose2d(image, weight=weight, bias=None, groups=batch, stride=self.stride, padding=self.padding)
        
        out = out.view(batch, output_nc, out.size(2), out.size(3))
        
        return out


# In[ ]:


class Noise(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
    
    def forward(self, image):
        noise = image.new_empty(image.size(0), 1, image.size(2), image.size(3)).normal_()
        result = image + self.weight * noise
        return result


# In[ ]:


class GeneratorBlock(nn.Module):
    def __init__(self, input_nc, output_nc, n_channel, dim_latent):
        super().__init__()
        
        self.conv1 = ModulatedConv2d(input_nc, output_nc, dim_latent, kernel_size=4, stride=2, padding=1).deconv()  # upsample
        self.noise1 = Noise()
        self.normalize1 = PixelwiseNormalization()
        self.activate1 = FReLU(output_nc)
        
        #self.attention = SelfAttention(output_nc)
        
        self.conv2 = ModulatedConv2d(output_nc, output_nc, dim_latent, kernel_size=3, stride=1, padding=1)
        self.noise2 = Noise()
        self.normalize2 = PixelwiseNormalization()
        self.activate2 = FReLU(output_nc)
        
        self.toRGB = WeightScaledConv2d(output_nc, n_channel, kernel_size=1, stride=1, padding=0)
        
    def forward(self, image, style):
        image = self.conv1(image, style)
        image = self.noise1(image)
        image = self.normalize1(image)
        image = self.activate1(image)
        
        #image = self.attention(image)
        
        image = self.conv2(image, style)
        image = self.noise2(image)
        image = self.normalize2(image)
        image = self.activate2(image)
        
        rgb = self.toRGB(image)
        
        return image, rgb


# In[ ]:


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        
        self.model = nn.Sequential(
            WeightScaledConv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1),  # downsample
            PixelwiseNormalization(),
            Mish(),
            #SelfAttention(output_nc),
            WeightScaledConv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1)
        )
        
        self.skip_conv = WeightScaledConv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)
            
        self.activation = nn.Sequential(
            PixelwiseNormalization(),
            Mish()
        )


    def forward(self, x):
        out = self.model(x)
        
        bilinear = F.interpolate(x, mode='bilinear', scale_factor=0.5, align_corners=True, recompute_scale_factor=True)
        skip = self.skip_conv(bilinear)
        
        out = out + skip
        out = self.activation(out)
        
        return out


# In[ ]:


class MappingNetwork(nn.Module):
    def __init__(self, dim_latent, num_depth):
        super().__init__()
        
        modules = []
        
        for _ in range(num_depth):
            modules += [WeightScaledLinear(dim_latent, dim_latent)]
            modules += [PixelwiseNormalization()]
            modules += [Mish()]
        
        self.module = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.module(x)
        return x


# In[ ]:


class Generator(nn.Module):
    def __init__(self, num_depth, num_fmap, num_mapping, small_size, n_channel=3):
        super().__init__()
        
        self.input_size = num_fmap(0)
        self.small_size = small_size
        self.register_buffer('const', torch.ones((1, self.input_size, 2, 2), dtype=torch.float32))
        
        self.style = MappingNetwork(self.input_size, num_mapping)
        self.blocks = nn.ModuleList([GeneratorBlock(num_fmap(i), num_fmap(i+1), n_channel, self.input_size) for i in range(num_depth)])
        
    def forward(self, styles, input_is_style=False):
        if not input_is_style:
            styles = [self.style(z) for z in styles]
        for _ in range(len(self.blocks) - len(styles)):
            styles += [styles[-1]]
        styles = [style.unsqueeze(1) for style in styles]
        styles = torch.cat(styles, dim=1).to(styles[0].device)
        
        x = self.const.expand(styles.size(0), self.input_size, 2, 2)
        
        prev_rgb = None
        for i, block in enumerate(self.blocks):
            x, rgb = block(x, styles[:,i,:])
            if prev_rgb is not None:
                upsampled = F.interpolate(prev_rgb, mode='bilinear', scale_factor=2, align_corners=True, recompute_scale_factor=True)
                rgb = rgb + upsampled  # Skip Connection
            prev_rgb = rgb
            if rgb.size(-1) == self.small_size:
                rgb_small = rgb
        
        rgb_small = torch.sigmoid(rgb_small)
        rgb = torch.sigmoid(rgb)
        
        return rgb, rgb_small, styles


# In[ ]:


class SimpleDecoder(nn.Module):
    class BasicBlock(nn.Module):
        def __init__(self, dim_in, dim_out):
            super().__init__()
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                WeightScaledConv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
                PixelwiseNormalization(),
                FReLU(dim_out)
            )
        def forward(self, x):
            return self.block(x)
    
    def __init__(self, input_nc, n_channel=3):
        super().__init__()
        self.block1 = SimpleDecoder.BasicBlock(input_nc, input_nc)
        self.block2 = SimpleDecoder.BasicBlock(input_nc, input_nc)
        self.block3 = SimpleDecoder.BasicBlock(input_nc, input_nc)
        self.block4 = SimpleDecoder.BasicBlock(input_nc, input_nc)
        
        self.toRGB = nn.Sequential(
            WeightScaledConv2d(input_nc, n_channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.toRGB(x)
        return x


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, num_depth, num_fmap, small_size, n_channel=3):
        super().__init__()
        
        self.small_size = small_size
        self.fromRGB = WeightScaledConv2d(n_channel, num_fmap(num_depth), kernel_size=1, stride=1, padding=0)
        self.fromRGB_small = nn.Conv2d(n_channel, num_fmap(np.log2(small_size)), kernel_size=1, stride=1, padding=0)
        self.blocks = nn.ModuleList([DiscriminatorBlock(num_fmap(i+1), num_fmap(i)) for i in range(num_depth)][::-1])
        
        self.decoder1 = SimpleDecoder(num_fmap(np.log2(small_size) - 3))
        self.decoder2 = SimpleDecoder(num_fmap(np.log2(small_size) - 4))
        self.mse_loss = nn.MSELoss()
        
        # PatchGAN
        self.conv_last = WeightScaledConv2d(num_fmap(0)+1, 1, kernel_size=3, stride=1, padding=1)
        
    def minibatch_standard_deviation(self, x):
        eps = 1e-8
        return torch.cat([x, torch.sqrt(((x - x.mean())**2).mean() + eps).expand(x.shape[0], 1, *x.shape[2:])], dim=1)
    
    def forward(self, x, is_real=False):
        if is_real:
            real_small = F.interpolate(x, size=(self.small_size, self.small_size))
            real_small_2 = F.interpolate(x, size=(self.small_size * 2, self.small_size * 2))
        else:
            x_small = x[1]
            x_small = self.fromRGB_small(x_small)
            x = x[0]
            
        x = self.fromRGB(x)
        
        for block in self.blocks:
            x = block(x)
            
            if is_real:
                if x.size(-1) * (2 ** 3) == self.small_size:
                    x_crop, pos = Util.randomCrop(x, self.small_size // 16)
                    fake_crop_small = self.decoder1(x_crop)
                if x.size(-1) * (2 ** 4) == self.small_size:
                    fake_small = self.decoder2(x)
            elif x.size(-1) <= self.small_size:
                x_small = block(x_small)
        
        x = self.minibatch_standard_deviation(x)
        out = self.conv_last(x)
        
        if is_real:
            real_crop_small = Util.crop(real_small_2, self.small_size, pos * 16)
            loss_recon = self.mse_loss(real_small, fake_small) + self.mse_loss(real_crop_small, fake_crop_small)
            return out, loss_recon
        else:
            x_small = self.minibatch_standard_deviation(x_small)
            out_small = self.conv_last(x_small)
            return out, out_small


# In[ ]:


class Util:
    @staticmethod
    def randomCrop(image, size):
        h = random.randrange(2) * size
        w = random.randrange(2) * size
        image = image[:, :, h:h+size, w:w+size]
        return image, (h, w)
    
    @staticmethod
    def crop(image, size, pos):
        image = image[:, :, pos[0]:pos[0]+size, pos[1]:pos[1]+size]
        return image
    
    @staticmethod
    def loadImages(batch_size, folder_path, size):
        imgs = ImageFolder(folder_path, transform=transforms.Compose([
            transforms.Resize(int(size)),
            transforms.RandomCrop(size),
            transforms.ToTensor()
        ]))
        return DataLoader(imgs, batch_size=batch_size, shuffle=True, drop_last=True)
    
    @staticmethod
    def augment(images):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.RandomErasing()
        ])
        device = images.device
        return torch.cat([transform(img).unsqueeze(0) for img in images.cpu()], 0).to(device)
    
    @staticmethod
    def showImage(image):
        get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib.pyplot as plt
        
        PIL = transforms.ToPILImage()
        ToTensor = transforms.ToTensor()
        
        img = PIL(image)
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(1, 1, 1) # (row, col, num)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img)
        #plt.gray()
        plt.show()
    
    @staticmethod
    def showImages(dataloader):
        for images in dataloader:
            for image in images[0]:
                Util.showImage(image)


# In[ ]:


class Solver:
    def __init__(self, args):
        use_cuda = torch.cuda.is_available() if not args.cpu else False
        self.device = torch.device("cuda" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        print(f'Use Device: {self.device}')
        
        def num_fmap(stage):
            base_size = self.args.image_size
            fmap_base = base_size * 4
            fmap_max = base_size // 2
            fmap_decay = 1.0
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        
        self.args = args
        self.feed_dim = num_fmap(0)
        self.max_depth = int(np.log2(self.args.image_size)) - 1
        small_size = args.image_size // 4
        
        self.netG = Generator(self.max_depth, num_fmap, self.args.num_mapping, small_size).to(self.device)
        self.netD = Discriminator(self.max_depth, num_fmap, small_size).to(self.device)
        self.state_loaded = False

        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)

        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=self.args.lr, betas=(0, 0.9))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.args.lr * self.args.mul_lr_dis, betas=(0, 0.9))
        #self.scheduler_G = CosineAnnealingLR(self.optimizer_G, T_max=4, eta_min=self.lr/4)
        #self.scheduler_D = CosineAnnealingLR(self.optimizer_D, T_max=4, eta_min=(self.lr * self.args.mul_lr_dis)/4)
        
        self.load_dataset()
        self.epoch = 0
    
    def weights_init(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(module.weight)
            if module.bias != None:
                module.bias.data.fill_(0)
            
    def load_dataset(self):
        self.dataloader = Util.loadImages(self.args.batch_size, self.args.image_dir, self.args.image_size)
        self.max_iters = len(iter(self.dataloader))
            
    def save_state(self, epoch):
        self.netG.cpu(), self.netD.cpu()
        torch.save(self.netG.state_dict(), os.path.join(self.args.weight_dir, f'weight_G.{epoch}.pth'))
        torch.save(self.netD.state_dict(), os.path.join(self.args.weight_dir, f'weight_D.{epoch}.pth'))
        self.netG.to(self.device), self.netD.to(self.device)
        
    def load_state(self):
        if (os.path.exists('weight_G.pth') and os.path.exists('weight_D.pth')):
            self.netG.load_state_dict(torch.load('weight_G.pth', map_location=self.device))
            self.netD.load_state_dict(torch.load('weight_D.pth', map_location=self.device))
            self.state_loaded = True
            print('Loaded network state.')
    
    def save_resume(self):
        with open(os.path.join('.', f'resume.pkl'), 'wb') as f:
            dump(self, f)
    
    def load_resume(self):
        if os.path.exists('resume.pkl'):
            with open(os.path.join('.', 'resume.pkl'), 'rb') as f:
                print('Load resume.')
                return load(f)
        else:
            return self
        
    def trainGAN(self, epoch, iters, max_iters, real_img, a=0, b=1, c=1):
        ### Train with LSGAN.
        ### for example, (a, b, c) = 0, 1, 1 or (a, b, c) = -1, 1, 0
        
        mse_loss = nn.MSELoss()
        style_feeds = [torch.randn(real_img.size(0), self.feed_dim).to(self.device)]
        noise = torch.Tensor(np.random.normal(0, self.args.lambda_zcr_noise, (real_img.size(0), self.feed_dim))).to(self.device)
        z = [style + noise for style in style_feeds]
        
        # ================================================================================ #
        #                             Train the discriminator                              #
        # ================================================================================ #
        
        # Compute loss with real images.
        real_img_aug = Util.augment(real_img)
        real_src_score, d_loss_recon = self.netD(real_img_aug, is_real=True)
        real_src_loss = torch.sum((real_src_score - b) ** 2)
        
        # Compute loss with fake images.
        fake_img, fake_img_small, _ = self.netG(style_feeds)
        fake_src_score, fake_small_score = self.netD((fake_img, fake_img_small))
        fake_src_loss = torch.sum((fake_src_score - a) ** 2)
        fake_small_loss = torch.sum((fake_small_score - a) ** 2)
        
        bcr_real = mse_loss(self.netD(real_img, is_real=True)[0], real_src_score)
        fake_img_aug = Util.augment(fake_img)
        fake_img_small_aug = Util.augment(fake_img_small)
        fake_src_score_aug, fake_small_score_aug = self.netD((fake_img_aug, fake_img_small_aug))
        bcr_fake = mse_loss(fake_src_score_aug, fake_src_score) + mse_loss(fake_small_score_aug, fake_small_score)
        
        z_img, z_small_img, _ = self.netG(z)
        z_score, z_small_score = self.netD((z_img, z_small_img))
        zcr_loss = mse_loss(fake_src_score, z_score) + mse_loss(fake_small_score, z_small_score)
        
        # Backward and optimize.
        d_loss = (0.5 * (real_src_loss + fake_src_loss + fake_small_loss) / self.args.batch_size + d_loss_recon
                  + self.args.lambda_bcr_real * bcr_real + self.args.lambda_bcr_fake * bcr_fake + self.args.lambda_zcr_dis * zcr_loss)
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
              
        # Logging.
        loss = {}
        loss['D/loss'] = d_loss.item()
        loss['D/bcr_loss'] = (bcr_real + bcr_fake).item()
        loss['D/zcr_loss'] = zcr_loss.item()
        
        # ================================================================================ #
        #                               Train the generator                                #
        # ================================================================================ #
        # Compute loss with reconstruction loss
        fake_img, fake_img_small, styles = self.netG(style_feeds)
        fake_src_score, fake_small_score = self.netD((fake_img, fake_img_small))
        fake_src_loss = torch.sum((fake_src_score - c) ** 2)
        fake_small_loss = torch.sum((fake_small_score - c) ** 2)
        
        z_img, z_small_img, _ = self.netG(z)
        zcr_loss = - mse_loss(fake_img, z_img) - mse_loss(fake_img_small, z_small_img)

        # Compute loss for path regularization
        noise = torch.randn_like(fake_img) / np.sqrt(fake_img.shape[2] * fake_img.shape[3])
        grad, = torch.autograd.grad(outputs=(fake_img * noise).sum(), inputs=styles, create_graph=True)
        path_length = grad.norm(2, dim=2).mean(1)
        path_mean = self.mean_path_length + 0.01 * (path_length.mean() - self.mean_path_length)
        path_penalty = (path_length - path_mean).square().mean()
        self.mean_path_length = path_mean.detach()

        # Backward and optimize.
        g_loss = 0.5 * (fake_src_loss + fake_small_loss) / self.args.batch_size + self.args.lambda_path * path_penalty + self.args.lambda_zcr_gen * zcr_loss
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        # Logging.
        loss['G/loss'] = g_loss.item()
        loss['G/path_penalty'] = path_penalty.item()
        loss['G/zcr_loss'] = zcr_loss.item()
        
        # Save
        if iters == max_iters:
            self.save_state(epoch)
            img_name = str(epoch) + '_' + str(iters) + '.png'
            img_path = os.path.join(self.args.result_dir, img_name)
            save_image(fake_img, img_path)
            #Util.showImage(fake_img)
        
        return loss
    
    def train(self, resume=True):
        self.netG.train()
        self.netD.train()
        
        while self.args.num_train > self.epoch:
            self.epoch += 1
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            
            self.mean_path_length = 0
            for iters, (data, _) in enumerate(tqdm(self.dataloader)):
                iters += 1
                
                data = data.to(self.device)
                
                loss = self.trainGAN(self.epoch, iters, self.max_iters, data)
                
                epoch_loss_D += loss['D/loss']
                epoch_loss_G += loss['G/loss']
                #experiment.log_metrics(loss)
                
            #self.scheduler_G.step()
            #self.scheduler_D.step()
            
            epoch_loss = epoch_loss_G + epoch_loss_D
            
            print(f'Epoch[{self.epoch}]'
                  #+ f' LR[G({self.scheduler_G.get_last_lr()[0]:.5f}) D({self.scheduler_D.get_last_lr()[0]:.5f})]'
                  + f' Loss[G({epoch_loss_G}) + D({epoch_loss_D}) = {epoch_loss}]')
                    
            if resume:
                self.save_resume()
    
    def generate(self, num=100):
        self.netG.eval()
        
        for _ in range(num):
            random_data = [torch.randn(1, self.netG.input_size).to(self.device)]
            fake_img = self.netG(random_data)[0][0,:]
            save_image(fake_img, os.path.join(self.args.result_dir, f'generated_{time.time()}.png'))
            #Util.showImage(fake_img)
        print('New picture was generated.')
        
    def showImages(self):
        Util.showImages(self.dataloader)


# In[ ]:


def main(args):
    hyper_params = {}
    hyper_params['Image Dir'] = args.image_dir
    hyper_params['Result Dir'] = args.result_dir
    hyper_params['Weight Dir'] = args.weight_dir
    hyper_params['Image Size'] = args.image_size
    hyper_params['Learning Rate'] = args.lr
    hyper_params["Mul Discriminator's LR"] = args.mul_lr_dis
    hyper_params['Batch Size'] = args.batch_size
    hyper_params['Num Train'] = args.num_train
    hyper_params['Num Mapping Net'] = args.num_mapping
    hyper_params['Path Regularize Coef'] = args.lambda_path
    hyper_params['bCR lambda_real'] = args.lambda_bcr_real
    hyper_params['bCR lambda_fake'] = args.lambda_bcr_fake
    hyper_params['zCR lambda_gen'] = args.lambda_zcr_gen
    hyper_params['zCR lambda_dis'] = args.lambda_zcr_dis
    hyper_params['zCR lambda_noise'] = args.lambda_zcr_noise
    
    solver = Solver(args)
    solver.load_state()
    
    if not args.noresume:
        solver = solver.load_resume()
    
    if args.generate > 0:
        solver.generate(args.generate)
        return
        
    for key in hyper_params.keys():
        print(f'{key}: {hyper_params[key]}')
    #experiment.log_parameters(hyper_params)
    
    #solver.showImages()
    solver.train(not args.noresume)
    #experiment.end()


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mul_lr_dis', type=float, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--num_mapping', type=int, default=8)
    parser.add_argument('--lambda_path', type=float, default=2)
    parser.add_argument('--lambda_bcr_real', type=float, default=10)
    parser.add_argument('--lambda_bcr_fake', type=float, default=10)
    parser.add_argument('--lambda_zcr_noise', type=float, default=0.07)
    parser.add_argument('--lambda_zcr_dis', type=float, default=20)
    parser.add_argument('--lambda_zcr_gen', type=float, default=0.5)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--generate', type=int, default=0)
    parser.add_argument('--noresume', action='store_true')

    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    
    main(args)

