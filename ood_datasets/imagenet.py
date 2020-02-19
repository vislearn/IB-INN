import torch
import numpy as np
import torch.utils.data

try:
    data = np.load('./ood_datasets/imagenet.npy')
    data = torch.from_numpy(data).float()
except FileNotFoundError:
    print('---'*40 + '\n')
    print('\trun "python -m ood_datasets.prepare_imagenet" first')
    print('\n' + '---'*40)
    raise

beta = torch.Tensor((0.4914, 0.4822, 0.4465)).view(-1, 1, 1)
gamma = 1. / torch.Tensor((0.247, 0.243, 0.261)).view(-1, 1, 1)

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

def imagenet(args):
    # TODO use the args (for tanh and padding mostly)
    aug = Augmentor(True, 0, beta, gamma, False)
    data_aug = torch.stack([aug(d) for d in data], dim=0)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_aug),
                           batch_size = 500,
                           num_workers = 1,
                           drop_last = False)

def imagenet_grayscale(args):
    # TODO use the args (for tanh and padding mostly)
    aug = Augmentor(True, 0, beta, gamma, False)
    data_aug = 0.2989 * data[:,0] + 0.5870 * data[:,1] + 0.1140 * data[:,2]
    data_aug = data_aug.view(-1, 1, 32, 32).expand(-1, 3, -1, -1)
    data_aug = torch.stack([aug(d) for d in data_aug], dim=0)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_aug),
                           batch_size = 500,
                           num_workers = 1,
                           drop_last = False)

def imagenet_noisy(args, noise_level=0.1):
    # TODO use the args (for tanh and padding mostly)
    aug = Augmentor(True, 0, beta, gamma, False)
    data_aug = data + noise_level * torch.randn_like(data)
    data_aug = data_aug.clamp(0., 1.)
    data_aug = torch.stack([aug(d) for d in data_aug], dim=0)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_aug),
                           batch_size = 500,
                           num_workers = 1,
                           drop_last = False)

# cifar noisy
# cifar grayscale
# cifar inverted colors
