from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsampleCouplingBlock(nn.Module):

    def __init__(self, dims_in, dims_c=[], subnet_constructor_strided=None,
                                           subnet_constructor_low_res=None,
                                           clamp=2.):
        super().__init__()

        channels = dims_in[0][0]
        if dims_c:
            raise ValueError('does not support conditioning yet')

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.splits = [self.split_len1, self.split_len2]

        self.in_channels = channels
        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.conditional = False
        condition_length = 0

        self.s_hi = subnet_constructor_strided(self.split_len1, 8 * self.split_len2)
        self.s_lo = subnet_constructor_low_res(4 * self.split_len2, self.split_len1 * 8)

        self.last_jac = None

        reshape_weights = torch.zeros(4,1,2,2)

        reshape_weights[0, 0, 0, 0] = 1
        reshape_weights[1, 0, 0, 1] = 1
        reshape_weights[2, 0, 1, 0] = 1
        reshape_weights[3, 0, 1, 1] = 1

        self.reshape_kernels = torch.nn.ParameterList()
        for split in self.splits:
            weights = torch.cat([reshape_weights] * split, 0)
            weights = nn.Parameter(weights)
            weights.requires_grad = False
            self.reshape_kernels.append(weights)

    def down(self, x, stream):
        return F.conv2d(x, self.reshape_kernels[stream], bias=None, stride=2, groups=self.splits[stream])

    def up(self, x, stream):
        return F.conv_transpose2d(x, self.reshape_kernels[stream], bias=None, stride=2, groups=self.splits[stream])

    def log_e(self, s):
        return self.clamp * torch.tanh(0.2 * s)

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
        split = [k for k in self.splits]
        if rev:
            split = [4*k for k in self.splits]

        x1, x2 = torch.split(x[0], split, dim=1)

        if not rev:
            y2 = self.down(x2, 1)
            a1 = self.s_hi(x1)
            y2, j2 = self.affine(y2, a1)

            y1 = self.down(x1, 0)
            a2 = self.s_lo(y2)
            y1, j1 = self.affine(y1, a2)

        else: # names of x and y are swapped!
            a2 = self.s_lo(x2)
            y1, j1 = self.affine(x1, a2, rev=True)
            y1 = self.up(y1, 0)

            a1 = self.s_hi(y1)
            y2, j2 = self.affine(x2, a1, rev=True)
            y2 = self.up(y2, 1)

        self.last_jac = j1 + j2
        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        output_dims = [k for k in input_dims[0]]
        output_dims[0] *= 4
        output_dims[1] //= 2
        output_dims[2] //= 2
        return [output_dims]

class HighPerfCouplingBlock(nn.Module):

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=2.):
        super().__init__()

        channels = dims_in[0][0]
        if dims_c:
            raise ValueError('does not support conditioning yet')

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.splits = [self.split_len1, self.split_len2]

        self.in_channels = channels
        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.conditional = False
        condition_length = 0

        self.s1 = subnet_constructor(self.split_len1, 2 * self.split_len2)
        self.s2 = subnet_constructor(self.split_len2, 2 * self.split_len1)

        self.last_jac = None

    def log_e(self, s):
        return self.clamp * torch.tanh(0.2 * s)

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
        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if not rev:
            a1 = self.s1(x1)
            y2, j2 = self.affine(x2, a1)

            a2 = self.s2(y2)
            y1, j1 = self.affine(x1, a2)

        else: # names of x and y are swapped!
            a2 = self.s2(x2)
            y1, j1 = self.affine(x1, a2, rev=True)

            a1 = self.s1(y1)
            y2, j2 = self.affine(x2, a1, rev=True)

        self.last_jac = j1 + j2
        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims

if __name__ == '__main__':

    import pdb

    import FrEIA.framework as Ff
    import FrEIA.modules as Fm

    def strided_constr(cin, cout):
        layers = [ nn.Conv2d(cin, 16, 3, stride=2, padding=1),
                   nn.ReLU(),
                   nn.Conv2d(16, cout, 1, stride=1)]
        return nn.Sequential(*layers)

    def low_res_constr(cin, cout):
        layers = [nn.Conv2d(cin, cout, 1)]
        return nn.Sequential(*layers)
    inp = Ff.InputNode(3, 32, 32, name='in')
    node = Ff.Node(inp, HighPerfCouplingBlock, {'subnet_constructor':low_res_constr}, name='coupling')
    node2 = Ff.Node(node, DownsampleCouplingBlock, {'subnet_constructor_strided':strided_constr,
                                                  'subnet_constructor_low_res':low_res_constr}, name='down_coupling')
    out = Ff.OutputNode(node2, name='out')

    net = Ff.ReversibleGraphNet([inp, node, node2, out])

    x = torch.randn(4, 3, 32, 32)

    z = net(x)
    jac = net.log_jacobian(run_forward=False)

    x_inv = net(z, rev=True)

    diff = x - x_inv
    print('shape in')
    print(x.shape)
    print('shape out')
    print(z.shape)
    print('shape inv')
    print(x_inv.shape)
    print('jacobian')
    print(jac)
    print('max abs difference')
    print(diff.abs().max())


