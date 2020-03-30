import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group

class AIO_Block(nn.Module):
    ''' Coupling block to replace the standard FrEIA implementation'''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor=None,
                 clamp=2.,
                 gin_block=False,
                 act_norm=1.,
                 permute_soft=False):

        super().__init__()

        channels = dims_in[0][0]
        if dims_c:
            raise ValueError('does not support conditioning yet')

        self.split_len1 = channels - channels // 2
        self.split_len2 = channels // 2
        self.splits = [self.split_len1, self.split_len2]

        self.n_pixels = dims_in[0][1] * dims_in[0][2]
        self.in_channels = channels
        self.clamp = clamp
        self.GIN = gin_block

        self.act_norm = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        self.act_offset = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        self.act_norm_trigger = True

        if act_norm:
            self.act_norm.data += np.log(act_norm)
            self.act_norm_trigger = False

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels,channels))
            for i,j in enumerate(np.random.permutation(channels)):
                w[i,j] = 1.
        w_inv = np.linalg.inv(w)

        self.w = nn.Parameter(torch.FloatTensor(w).view(channels, channels, 1, 1),
                              requires_grad=False)
        self.w_inv = nn.Parameter(torch.FloatTensor(w_inv).view(channels, channels, 1, 1),
                              requires_grad=False)

        self.conditional = False
        condition_length = 0

        self.s = subnet_constructor(self.split_len1, 2 * self.split_len2)
        self.last_jac = None

    def log_e(self, s):
        s = self.clamp * torch.tanh(0.1 * s)
        if self.GIN:
            s -= torch.mean(s, dim=(1,2,3), keepdim=True)
        return s

    def permute(self, x, rev=False):
        if rev:
            return F.conv2d((x - self.act_offset) * (-self.act_norm).exp(), self.w_inv)
        else:
            return F.conv2d(x, self.w) * self.act_norm.exp() + self.act_offset

    def affine(self, x, a, rev=False):
        ch = x.shape[1]
        sub_jac = self.log_e(a[:,:ch])
        if not rev:
            return (x * torch.exp(sub_jac) + a[:,ch:],
                    torch.sum(sub_jac, dim=(1,2,3)))
        else:
            return ((x - a[:,ch:]) * torch.exp(-sub_jac),
                    -torch.sum(sub_jac, dim=(1,2,3)))

    def forward(self, x, c=[], rev=False):
        if self.act_norm_trigger:
            with torch.no_grad():
                print('ActNorm triggered')
                self.act_norm_trigger = False
                x_out = self.forward(x)[0]
                x_out = x_out.transpose(0,1).contiguous().view(self.in_channels, -1)
                self.act_norm.data -= x_out.std(dim=1, unbiased=False).log().view(1, self.in_channels, 1, 1)
                self.act_offset.data -= x_out.mean(dim=1).view(1, self.in_channels, 1, 1)

        if rev:
            x = [self.permute(x[0], rev=True)]

        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if not rev:
            a1 = self.s(x1)
            x2, j2 = self.affine(x2, a1)
        else: # names of x and y are swapped!
            a1 = self.s(x1)
            x2, j2 = self.affine(x2, a1, rev=True)

        self.last_jac = j2
        x_out = torch.cat((x1, x2), 1)

        if not rev:
            x_out = self.permute(x_out, rev=False)

        return [x_out]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac + (-1)**rev * self.act_norm.sum() * self.n_pixels

    def output_dims(self, input_dims):
        return input_dims

if __name__ == '__main__':
    N = 8
    c = 48
    x = torch.FloatTensor(128, c, N, N)
    x.normal_(0,1)

    constr = lambda c_in, c_out: torch.nn.Conv2d(c_in, c_out, 1)

    layer = AIO_Block([(c, N, N)],
                 subnet_constructor=constr,
                 clamp=2.,
                 gin_block=False,
                 act_norm=0,
                 permute_soft=True)

    transf = layer([x])

    transf = layer([x])
    x_inv = layer(transf, rev=True)[0]

    err = torch.abs(x - x_inv)
    print(err.max().item())
    print(err.mean().item())


