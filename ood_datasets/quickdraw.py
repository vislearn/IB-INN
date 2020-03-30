import torch
import numpy as np
import torch.utils.data

files = ['ood_datasets/car.npy',
         'ood_datasets/cat.npy',
         'ood_datasets/dog.npy',
         'ood_datasets/airplane.npy']

data = np.concatenate([np.load(f)[:2000] for f in files], axis=0)
data = data.reshape((-1, 1, 28, 28 ))
data = data.astype(np.float32) / 255.
np.random.seed(0)
np.random.shuffle(data)
color_vector = 0.3 + 0.7 * np.random.random((data.shape[0], 3, 1, 1)).astype(np.float32)
data = data * color_vector
data = 1. - data
data = np.pad(data, ((0,0), (0,0), (2,2), (2,2)), mode='edge')

np.random.seed(None)

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

def quickdraw_colored(args):
    aug = Augmentor(True, 0, beta, gamma, False)
    data_aug = torch.stack([aug(d) for d in torch.from_numpy(data)], dim=0)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_aug),
                           batch_size = 500,
                           num_workers = 1,
                           drop_last = False)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(32, 32))
    for k in range(100):
        plt.subplot(10, 10, k+1)
        plt.imshow(data[k].transpose((1,2,0)))
        plt.xticks([]); plt.yticks([])

    plt.tight_layout()
    plt.savefig('ood_datasets/quickdraw_examples.pdf')
