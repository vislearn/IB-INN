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

        self.test_augmentor =  Augmentor(True,  0.,    beta, gamma, tanh, channel_pad, channel_pad_sigma)
        self.transform = T.Compose([T.ToTensor(), self.test_augmentor])

        self.dims = (3 + channel_pad, 32, 32)
        self.channels = 3 + channel_pad

        data_dir = 'svhn_data'
        self.n_classes = 10
        dataset_class = torchvision.datasets.SVHN

        self.test_data = dataset_class(data_dir, split='test', download=True,
                                               transform=T.Compose([T.ToTensor(), self.test_augmentor]))

        self.test_loader   = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                                   num_workers=2, pin_memory=True, drop_last=True)

    def de_augment(self, x):
        return self.test_augmentor.de_augment(x)

    def augment(self, x):
        return self.test_augmentor(x)


def svhn(args):
    return Dataset(args).test_loader
