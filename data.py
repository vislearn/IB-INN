import os
from os.path import join, isfile, basename
from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets

class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)

class LabelAugmentor():
    def __init__(self, mapping=list(range(10))):
        self.mapping = mapping

    def __call__(self, l):
        return int(self.mapping[l])

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

        if isinstance(self.gamma, float):
            return x / self.gamma + self.beta
        else:
            return x / self.gamma.to(x.device) + self.beta.to(x.device)

class Dataset():

    def __init__(self, args):

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

        self.train_augmentor = Augmentor(False, noise, beta, gamma, tanh, channel_pad, channel_pad_sigma)
        self.test_augmentor =  Augmentor(True,  0.,    beta, gamma, tanh, channel_pad, channel_pad_sigma)
        self.transform = T.Compose([T.ToTensor(), self.test_augmentor])

        if self.dataset == 'MNIST':
            self.dims = (28, 28)
            if channel_pad:
                raise ValueError('needs to be fixed, channel padding does not work with mnist')
            self.channels = 1
            self.n_classes = 10
            self.label_mapping = list(range(self.n_classes))
            self.label_augment = LabelAugmentor(self.label_mapping)
            data_dir = 'mnist_data'

            self.test_data = torchvision.datasets.MNIST(data_dir, train=False, download=True,
                                                   transform=T.Compose([T.ToTensor(), self.test_augmentor]),
                                                   target_transform=self.label_augment)
            self.train_data = torchvision.datasets.MNIST(data_dir, train=True, download=True,
                                                    transform=T.Compose([T.ToTensor(), self.train_augmentor]),
                                                    target_transform=self.label_augment)
        elif self.dataset in ['CIFAR10', 'CIFAR100']:
            self.dims = (3 + channel_pad, 32, 32)
            self.channels = 3 + channel_pad

            if self.dataset == 'CIFAR10':
                data_dir = 'cifar_data'
                self.n_classes = 10
                dataset_class = torchvision.datasets.CIFAR10
            else:
                data_dir = 'cifar100_data'
                self.n_classes = 100
                dataset_class = torchvision.datasets.CIFAR100

            self.label_mapping = list(range(self.n_classes))
            self.label_augment = LabelAugmentor(self.label_mapping)

            self.test_data = dataset_class(data_dir, train=False, download=True,
                                                   transform=T.Compose([T.ToTensor(), self.test_augmentor]),
                                                   target_transform=self.label_augment)
            self.train_data = dataset_class(data_dir, train=True, download=True,
                                                   transform=T.Compose([T.RandomHorizontalFlip(),
                                                                       T.ColorJitter(0.1, 0.1, 0.05),
                                                                       T.Pad(8, padding_mode='edge'),
                                                                       T.RandomRotation(12),
                                                                       T.CenterCrop(36),
                                                                       T.RandomCrop(32),
                                                                       T.ToTensor(),
                                                                       self.train_augmentor]),
                                                    target_transform=self.label_augment)

        else:
            raise ValueError(f"what is this dataset, {args['data']['dataset']}?")

        self.train_data, self.val_data = torch.utils.data.random_split(self.train_data, (len(self.train_data) - 1024, 1024))

        self.val_x = torch.stack([x[0] for x in self.val_data], dim=0).cuda()
        self.val_y = self.onehot(torch.LongTensor([x[1] for x in self.val_data]).cuda(), label_smoothing)

        self.train_loader  = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                                   num_workers=10, pin_memory=True, drop_last=True)
        self.test_loader   = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                                   num_workers=6, pin_memory=True, drop_last=True)

    def show_data_hist(self):
        x = self.val_x.cpu().numpy()
        plt.hist(x.flatten(), bins=200)
        plt.show()

    def de_augment(self, x):
        return self.test_augmentor.de_augment(x)

    def augment(self, x):
        return self.test_augmentor(x)

    def onehot(self, l, label_smooth=0):
        y = torch.cuda.FloatTensor(l.shape[0], self.n_classes).zero_()
        y.scatter_(1, l.view(-1, 1), 1.)
        if label_smooth:
            y = y * (1 - label_smooth) + label_smooth / self.n_classes
        return y
