import os
from os.path import join, isfile, basename
from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import conv2d
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.datasets

class Augmentor():
    def __init__(self, deterministic, noise_amplitde, beta, gamma, tanh, ch_pad=0, ch_pad_sig=0):
        self.deterministic = deterministic
        self.sigma_noise = noise_amplitde
        self.beta = beta
        self.gamma = gamma
        self.tanh = tanh
        self.ch_pad = ch_pad
        self.ch_pad_sig = ch_pad_sig
        assert ch_pad_sig <= 1., 'Padding sigma must be between 0 and 1.'

    def __call__(self, x):
        if not self.deterministic:
            x += torch.rand_like(x) / 256.
            if self.sigma_noise > 0.:
                x += self.sigma_noise * torch.randn_like(x)

        x = self.gamma * (x - self.beta)

        if self.tanh:
            x.clamp_(min=-(1 - 1e-7), max=(1 - 1e-7))
            x = 0.5 * torch.log((1+x) / (1-x))

        if self.ch_pad:
            padding = torch.cat([x] * int(np.ceil(float(self.ch_pad) / x.shape[0])), dim=0)[:self.ch_pad]
            padding *= np.sqrt(1. - self.ch_pad_sig**2)
            padding += self.ch_pad_sig * torch.randn(self.ch_pad, x.shape[1], x.shape[2])
            x = torch.cat([x, padding], dim=0)

        return x

    def de_augment(self, x):
        if self.ch_pad:
            x = x[:, :-self.ch_pad]

        if self.tanh:
            x = torch.tanh(x)

        return x / self.gamma.to(x.device) + self.beta.to(x.device)

class Dataset():

    def __init__(self, args, extra_transforms):

        self.dataset = args['data']['dataset']
        self.batch_size = int(args['data']['batch_size'])
        tanh = eval(args['data']['tanh_augmentation'])
        noise = float(args['data']['noise_amplitde'])
        label_smoothing = float(args['data']['label_smoothing'])
        channel_pad = int(args['data']['pad_noise_channels'])
        channel_pad_sigma = float(args['data']['pad_noise_std'])

        if self.dataset == 'MNIST':
            beta = 0.5
            gamma = 2.
        else:
            beta = torch.Tensor((0.4914, 0.4822, 0.4465)).view(-1, 1, 1)
            gamma = 1. / torch.Tensor((0.247, 0.243, 0.261)).view(-1, 1, 1)

        self.test_augmentor =  Augmentor(True,  0.,    beta, gamma, tanh, channel_pad, channel_pad_sigma)
        self.transform = T.Compose([T.ToTensor(), self.test_augmentor])

        self.dims = (3 + channel_pad, 32, 32)
        self.channels = 3 + channel_pad

        if self.dataset == 'CIFAR10':
            data_dir = 'cifar_data'
            self.n_classes = 10
            dataset_class = torchvision.datasets.CIFAR10
        elif self.dataset == 'CIFAR100':
            data_dir = 'cifar100_data'
            self.n_classes = 100
            dataset_class = torchvision.datasets.CIFAR100
        else:
            raise ValueError("Only CIFAR10 and CIFAR100 supported for OoD datasets")

        self.test_data = dataset_class(data_dir, train=False, download=True,
                                               transform=T.Compose(extra_transforms + [self.test_augmentor]))

        self.test_loader   = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                                   num_workers=2, pin_memory=True, drop_last=True)

    def de_augment(self, x):
        return self.test_augmentor.de_augment(x)

    def augment(self, x):
        return self.test_augmentor(x)


def cifar_flipped(args):
    return Dataset(args, [T.RandomVerticalFlip(1.0), T.ToTensor()]).test_loader

def cifar_hue(args, alpha):
    return Dataset(args, [lambda x: F.adjust_hue(x, alpha/2), T.ToTensor()]).test_loader

def cifar_rgb_rotation(args, alpha):
    a = np.pi * (0.2 * alpha ** 0.7)
    b = np.pi * (1.0 * alpha ** 1.2)
    scale = 1 + 0.0 * alpha

    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0,1,0], [-np.sin(b), 0, np.cos(b)]])
    rotation_kernel = scale * np.dot(rx, ry)
    rotation_kernel = torch.from_numpy(rotation_kernel).view(3, 3, 1, 1).float()
    def rotate_rgb(x):
        x = x.expand(1, -1, -1, -1) - 0.5
        x = conv2d(x, rotation_kernel)
        return (x + 0.5).clamp(0.,1.).squeeze()

    return Dataset(args, [T.ToTensor(), rotate_rgb]).test_loader


def cifar_noise(args, noise_level=0.1):
    def add_noise(x):
        x = x + noise_level * torch.randn_like(x)
        return x.clamp(0., 1.)

    return Dataset(args, [T.ToTensor(), add_noise]).test_loader

beta = torch.Tensor((0.4914, 0.4822, 0.4465)).view(-1, 1, 1)
gamma = 1. / torch.Tensor((0.247, 0.243, 0.261)).view(-1, 1, 1)
default_augment = Augmentor(True, 0, beta, gamma, False, ch_pad=0, ch_pad_sig=0)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import configparser

    default_args = configparser.ConfigParser()
    default_args.read('default.ini')

    w = 8
    h = 1
    dh = 1
    plt.figure(figsize=(2*w, 2*h))
    for i, a in enumerate(np.linspace(0, 1, w)):
        loader = cifar_rgb_rotation(default_args, a)
        for x,y in loader:
            x = default_augment.de_augment(x)
            for j in range(h):
                plt.subplot(h, w, 1 + w*j + i)
                plt.imshow(x[j+dh].numpy().transpose((1,2,0)))
                plt.xticks([])
                plt.yticks([])
            break

    plt.tight_layout()
    plt.savefig('figures/data_ood_interpolation.png', dpi=120)
