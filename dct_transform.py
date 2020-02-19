import numpy as np
import torch
import torch.nn as nn

'''adapted from https://github.com/zh217/torch-dct'''

def dct_1d(x):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    :param x: the input signal
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = 2 * V.view(*x_shape)

    return V

def idct_1d(X):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x

    :param X: the input signal
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.irfft(V, 1, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

class DCTPooling2d(nn.Module):
    def __init__(self, dims_in, rebalance=1.):
        super().__init__()
        self.ch = dims_in[0][0]
        self.N = dims_in[0][1]
        self.rebalance = 2 * (self.N +1) / rebalance
        self.jac = (self.N**2 * self.ch) * np.log(rebalance)

        assert torch.cuda.is_available(), "please father, give 1 cuda"
        I = torch.eye(self.N).cuda()

        self.weight = dct_1d(I).t()
        self.inv_weight = idct_1d(I).t()

        self.weight = nn.Parameter(self.weight, requires_grad=False)
        self.inv_weight = nn.Parameter(self.inv_weight, requires_grad=False)

    def forward(self, x, rev=False):
        x = x[0]

        if rev:
            weight = self.inv_weight
            rebal = self.rebalance
            x = x.view(x.shape[0], self.N, self.N, self.ch)
            x = x.transpose(1, -1).contiguous()
        else:
            weight = self.weight
            rebal = 1/self.rebalance

        out = nn.functional.linear(x, weight)
        out = nn.functional.linear(out.transpose(-1, -2), weight)
        out = out.transpose(-1, -2)

        if not rev:
            out = out.transpose(1, -1).contiguous()
            out = out.view(out.shape[0], -1)

        return [out * rebal]

    def jacobian(self, x, rev=False):
        return self.jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        comp = c*w*h
        return [(comp,)]

if __name__ == '__main__':
    for N in [16, 32, 64]:
        x = torch.cuda.FloatTensor(1000, 3, N, N)
        x.normal_(0,1)
        dct_layer = DCTPooling2d([(3, N, N)])
        transf = dct_layer(x)
        x_inv = dct_layer(transf, rev=True)

        transf = transf.contiguous()

        means = transf[:, 3:6]
        true_means = torch.mean(x, dim=(2,3))

        err = torch.abs(x - x_inv).max()
        print(N, err.item(), flush=True)


