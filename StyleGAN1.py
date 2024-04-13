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


# Adaptive Instance Normalization
class AdaIn(nn.Module):
    def __init__(self, n_channel, dim_latent):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)
        self.transform = WeightScaledLinear(dim_latent, n_channel * 2, bias=True)
        self.transform.bias.data[n_channel:] = 0
        self.transform.bias.data[:n_channel] = 1
        
    def forward(self, image, style):
        factor, bias = self.transform(style).unsqueeze(2).unsqueeze(3).chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result


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
    def __init__(self, input_nc, output_nc, num_channels, dim_latent):
        super().__init__()
        
        self.conv1 = WeightScaledConv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1).deconv()  # upsample
        self.noise1 = Noise()
        self.adain1 = AdaIn(output_nc, dim_latent)
        self.normalize1 = PixelwiseNormalization()
        self.activate1 = FReLU(output_nc)
        
        #self.attention = SelfAttention(output_nc)
        
        self.conv2 = WeightScaledConv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1)
        self.noise2 = Noise()
        self.adain2 = AdaIn(output_nc, dim_latent)
        self.normalize2 = PixelwiseNormalization()
        self.activate2 = FReLU(output_nc)
        
        self.toRGB = WeightScaledConv2d(output_nc, num_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, image, style, last=False):
        image = self.conv1(image)
        image = self.noise1(image)
        image = self.adain1(image, style)
        image = self.normalize1(image)
        image = self.activate1(image)
        
        #image = self.attention(image)
        
        image = self.conv2(image)
        image = self.noise2(image)
        image = self.adain2(image, style)
        image = self.normalize2(image)
        image = self.activate2(image)
        
        if last:
            image = self.toRGB(image)
        return image


# In[ ]:


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_nc, output_nc, num_channels):
        super().__init__()
        
        self.fromRGB = WeightScaledConv2d(num_channels, input_nc, kernel_size=1, stride=1, padding=0)
        self.model = nn.Sequential(
            WeightScaledConv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1),  # downsample
            PixelwiseNormalization(),
            Mish(),
            #SelfAttention(output_nc),
            WeightScaledConv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1),
            PixelwiseNormalization(),
            Mish()
        )

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        x = self.model(x)
        return x


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
    def __init__(self, num_depth, num_channels, num_fmap, num_mapping):
        super().__init__()
        
        self.input_size = num_fmap(0)
        self.register_buffer('const', torch.ones((1, self.input_size, 2, 2), dtype=torch.float32))
        self.style = MappingNetwork(num_fmap(0), num_mapping)
        self.blocks = nn.ModuleList([GeneratorBlock(num_fmap(i), num_fmap(i+1), num_channels, num_fmap(0)) for i in range(num_depth)])
        
        self.depth = 0
        self.alpha = 1.0

    def forward(self, styles, input_is_style=False):
        if not input_is_style:
            styles = [self.style(z) for z in styles]
        for _ in range(len(self.blocks) - len(styles)):
            styles += [styles[-1]]
        
        x = self.const.expand(styles[0].size(0), self.input_size, 2, 2)
        rgb = x = self.blocks[0](x, styles[0], self.depth == 0)
        
        if self.depth > 0:
            for i in range(self.depth - 1):
                x = self.blocks[i+1](x, styles[i+1])
            rgb = self.blocks[self.depth](x, styles[self.depth], last=True)
            if self.alpha < 1.0:
                prev_rgb = self.blocks[self.depth - 1].toRGB(x)
                prev_rgb = F.interpolate(prev_rgb, mode='bilinear', scale_factor=2, align_corners=True, recompute_scale_factor=True)
                rgb = (1 - self.alpha) * prev_rgb + self.alpha * rgb
        
        rgb = torch.sigmoid(rgb)
        
        return rgb, styles


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, num_depth, num_channels, num_fmap):
        super().__init__()

        self.blocks = nn.ModuleList([DiscriminatorBlock(num_fmap(i+1), num_fmap(i), num_channels) for i in range(num_depth)][::-1])

        # PatchGAN
        self.conv_last = WeightScaledConv2d(num_fmap(0)+1, 1, kernel_size=3, stride=1, padding=1)
        
        self.depth = 0
        self.alpha = 1.0
        
    def minibatch_standard_deviation(self, x):
        eps = 1e-8
        return torch.cat([x, torch.sqrt(((x - x.mean())**2).mean() + eps).expand(x.shape[0], 1, *x.shape[2:])], dim=1)
    
    def forward(self, x):
        out = self.blocks[-(self.depth + 1)](x, first=True)
        
        if self.depth > 0 and self.alpha < 1.0:
            x = F.interpolate(x, mode='bilinear', scale_factor=0.5, align_corners=True, recompute_scale_factor=True)
            prev = self.blocks[-self.depth].fromRGB(x)
            out = self.alpha * out + (1 - self.alpha) * prev
                
        for i in range(self.depth, 0, -1):
            out = self.blocks[-i](out)
        
        out = self.minibatch_standard_deviation(out)
        out = self.conv_last(out)
        
        return out


# In[ ]:


class Util:
    @staticmethod
    def loadImages(batch_size, folder_path, size):
        imgs = ImageFolder(folder_path, transform=transforms.Compose([
            transforms.Resize(int(size)),
            #transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size),
            #transforms.RandomRotation(degrees=30),
            #transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5),
            transforms.ToTensor()
        ]))
        return DataLoader(imgs, batch_size=batch_size, shuffle=True, drop_last=True)
    
    @staticmethod
    def showImages(dataloader):
        get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib.pyplot as plt
        
        PIL = transforms.ToPILImage()
        ToTensor = transforms.ToTensor()

        for images in dataloader:
            for image in images[0]:
                img = PIL(image)
                fig = plt.figure(dpi=200)
                ax = fig.add_subplot(1, 1, 1) # (row, col, num)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(img)
                #plt.gray()
                plt.show()


# In[ ]:


class Solver:
    def __init__(self, use_cpu, lr, mul_lr_dis, batch_size_base, img_dir, image_size, num_train_base, num_mapping, result_dir, weight_dir):
        use_cuda = torch.cuda.is_available() if not use_cpu else False
        self.device = torch.device("cuda" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        print(f'Use Device: {self.device}')
        
        def num_fmap(stage):
            base_size = self.image_size
            fmap_base = base_size * 4
            fmap_max = base_size // 2
            fmap_decay = 1.0
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        
        self.lr = lr
        self.batch_size_base = batch_size_base
        self.image_size = image_size
        self.num_train_base = num_train_base
        self.num_mapping = num_mapping
        self.result_dir = result_dir
        self.weight_dir = weight_dir
        self.img_dir = img_dir
        self.image_size = image_size
        self.num_channels = 3
        self.feed_dim = num_fmap(0)
        self.max_depth = int(np.log2(self.image_size)) - 1
        self.depth = 0
        
        self.netG = Generator(self.max_depth, self.num_channels, num_fmap, self.num_mapping).to(self.device)
        self.netD = Discriminator(self.max_depth, self.num_channels, num_fmap).to(self.device)
        self.state_loaded = False

        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)
        
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0, 0.9))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.lr * mul_lr_dis, betas=(0, 0.9))
        #self.scheduler_G = CosineAnnealingLR(self.optimizer_G, T_max=4, eta_min=self.lr/4)
        #self.scheduler_D = CosineAnnealingLR(self.optimizer_D, T_max=4, eta_min=(self.lr * mul_lr_dis)/4)
        
        self.epoch = 0
        self.num_train = 0
    
    def weights_init(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(module.weight)
            if module.bias:
                module.bias.data.fill_(0)
            
    def load_dataset(self):
        self.batch_size  = self.batch_size_base * (self.max_depth - self.depth)
        image_size = self.image_size / 2 ** (self.max_depth - self.depth - 1)
        self.dataloader = Util.loadImages(self.batch_size, self.img_dir, image_size)
        self.max_iters = len(iter(self.dataloader))
            
    def save_state(self, epoch):
        self.netG.cpu(), self.netD.cpu()
        torch.save(self.netG.state_dict(), os.path.join(self.weight_dir, f'weight_G.{epoch}.pth'))
        torch.save(self.netD.state_dict(), os.path.join(self.weight_dir, f'weight_D.{epoch}.pth'))
        #with open(os.path.join(self.weight_dir, f'weight_G.{epoch}.pkl'), 'wb') as f:
        #    dump(self.netG, f)
        #with open(os.path.join(self.weight_dir, f'weight_D.{epoch}.pkl'), 'wb') as f:
        #    dump(self.netD, f)
        self.netG.to(self.device), self.netD.to(self.device)
        
    def load_state(self):
        if (os.path.exists('weight_G.pth') and os.path.exists('weight_D.pth')):
            self.netG.load_state_dict(torch.load('weight_G.pth', map_location=self.device))
            self.netD.load_state_dict(torch.load('weight_D.pth', map_location=self.device))
            #with open(os.path.join('.', f'weight_G.pkl'), 'rb') as f:
            #    self.netG = load(f)
            #with open(os.path.join('.', f'weight_D.pkl'), 'rb') as f:
            #    self.netD = load(f)
            #self.netG = self.netG.to(self.device)
            #self.netD = self.netD.to(self.device)
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
        
        style_feeds = [torch.randn(real_img.size(0), self.feed_dim).to(self.device)]
        
        # ================================================================================ #
        #                             Train the discriminator                              #
        # ================================================================================ #
        
        # Compute loss with real images.
        real_src_score = self.netD(real_img)
        real_src_loss = torch.sum((real_src_score - b) ** 2)
        
        # Compute loss with fake images.
        fake_img, _ = self.netG(style_feeds)
        fake_src_score = self.netD(fake_img)
        fake_src_loss = torch.sum((fake_src_score - a) ** 2)
        
        # Backward and optimize.
        d_loss = 0.5 * (real_src_loss + fake_src_loss) / self.batch_size
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
              
        # Logging.
        loss = {}
        loss['D/loss'] = d_loss.item()
        
        # ================================================================================ #
        #                               Train the generator                                #
        # ================================================================================ #
        # Compute loss with reconstruction loss
        fake_img, _ = self.netG(style_feeds)
        fake_src_score = self.netD(fake_img)
        fake_src_loss = torch.sum((fake_src_score - c) ** 2)

        # Backward and optimize.
        g_loss = 0.5 * fake_src_loss / self.batch_size
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        # Logging.
        loss['G/loss'] = g_loss.item()
        
        # Save
        if iters == max_iters:
            self.save_state(epoch)
            img_name = str(epoch) + '_' + str(iters) + '.png'
            img_path = os.path.join(self.result_dir, img_name)
            save_image(fake_img, img_path)
        
        return loss
    
    def train(self, resume=True):
        self.netG.train()
        self.netD.train()
        self.load_dataset()
        
        while True:
            #max_train = self.num_train_base
            max_train = int(self.num_train_base // np.log2(self.max_depth - self.depth + 1))
            self.num_train += 1
            self.epoch += 1
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            
            for iters, (data, _) in enumerate(tqdm(self.dataloader)):
                iters += 1
                
                alpha = self.num_train / max_train
                alpha = min(1.0, alpha * 2)
                self.netG.alpha = self.netD.alpha = alpha
                
                data = data.to(self.device)
                
                loss = self.trainGAN(self.epoch, iters, self.max_iters, data)
                
                epoch_loss_D += loss['D/loss']
                epoch_loss_G += loss['G/loss']
                #experiment.log_metrics(loss)
            
            epoch_loss = epoch_loss_G + epoch_loss_D
            
            print(f'Epoch[{self.epoch}] Depth[{self.depth+1}/{self.max_depth}]'
                  + f' DepthTrain[{self.num_train}/{max_train}] BatchSize[{self.batch_size}]'
                  #+ f' LR[G({self.scheduler_G.get_last_lr()[0]:.5f}) D({self.scheduler_D.get_last_lr()[0]:.5f})]'
                  + f' Loss[G({epoch_loss_G}) + D({epoch_loss_D}) = {epoch_loss}]')
                
            #self.scheduler_G.step()
            #self.scheduler_D.step()
            
            if self.num_train >= max_train:
                if self.depth+1 < self.max_depth:
                    self.depth += 1
                    self.netG.depth = self.netD.depth = self.depth
                    self.load_dataset()  # Change batch-size and image-size.
                    self.num_train = 0
                else:
                    break
                    
            if resume:
                self.save_resume()
    
    def generate(self, num=100):
        self.netG.eval()
        
        for _ in range(num):
            random_data = [torch.randn(1, self.netG.input_size).to(self.device)]
            fake_img = self.netG(random_data)[0].cpu().data[0]
            save_image(fake_img, os.path.join(self.result_dir, f'generated_{time.time()}.png'))
            
        print('New picture was generated.')
        
    def showImages(self):
        depth = self.depth
        self.depth = self.max_depth - 1
        self.load_dataset()
        Util.showImages(self.dataloader)
        self.depth = depth
        self.load_dataset()


# In[ ]:


def main(args):
    hyper_params = {}
    hyper_params['Img Dir'] = args.img_dir
    hyper_params['Result Dir'] = args.result_dir
    hyper_params['Weight Dir'] = args.weight_dir
    hyper_params['Img Size'] = args.img_size
    hyper_params['Learning Rate'] = args.lr
    hyper_params["Mul Discriminator's LR"] = args.mul_lr_dis
    hyper_params['Batch Size Base'] = args.batch_size_base
    hyper_params['Num Train Base'] = args.num_train_base
    hyper_params['Num Mapping Net'] = args.num_mapping
    
    solver = Solver(args.cpu, args.lr, args.mul_lr_dis, args.batch_size_base, args.img_dir, args.img_size, args.num_train_base,
                    args.num_mapping, args.result_dir, args.weight_dir)
    solver.load_state()
    
    if not args.noresume:
        solver = solver.load_resume()
    
    hyper_params['Max Depth'] = solver.max_depth
    hyper_params['Start Depth'] = args.start_depth
    
    if args.start_depth != 1:
        solver.netG.depth = solver.netD.depth = solver.depth = args.start_depth - 1
    
    if args.generate > 0:
        solver.generate(args.generate)
        return
        
    for key in hyper_params.keys():
        print(f'{key}: {hyper_params[key]}')
    #experiment.log_parameters(hyper_params)
    
    #solver.showImages()
    solver.train(not args.noresume)


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mul_lr_dis', type=float, default=4)
    parser.add_argument('--batch_size_base', type=int, default=8)
    parser.add_argument('--num_train_base', type=int, default=100)
    parser.add_argument('--num_mapping', type=int, default=8)
    parser.add_argument('--start_depth', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--generate', type=int, default=0)
    parser.add_argument('--noresume', action='store_true')

    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
        
    main(args)

