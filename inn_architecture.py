from functools import partial
import pdb

import torch
import torch.nn as nn
from torch.nn.functional import conv2d, interpolate
import numpy as np

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from downsampling_coupling_block import DownsampleCouplingBlock
from all_in_one_block import AIO_Block
from dct_transform import DCTPooling2d

def construct_irevnet(classifier):
    inn = constuct_inn(classifier)
    projection_layer = nn.Linear(classifier.ndim_tot, classifier.n_classes)

    class RevNetWrapper(nn.Module):
        def __init__(self, inn, projection_layer):
            super().__init__()
            self.inn = inn
            self.proj = projection_layer

        def forward(self, x):
            return self.proj(self.inn(x))

    return RevNetWrapper(inn, projection_layer)

def constuct_inn(classifier, verbose=False):

    fc_width = int(classifier.args['model']['fc_width'])
    n_coupling_blocks_fc = int(classifier.args['model']['n_coupling_blocks_fc'])

    use_dct = eval(classifier.args['model']['dct_pooling'])
    conv_widths = eval(classifier.args['model']['conv_widths'])
    n_coupling_blocks_conv = eval(classifier.args['model']['n_coupling_blocks_conv'])
    dropouts = eval(classifier.args['model']['dropout_conv'])
    dropouts_fc = float(classifier.args['model']['dropout_fc'])

    groups = int(classifier.args['model']['n_groups'])
    clamp = float(classifier.args['model']['clamp'])

    ndim_input = classifier.dims

    batchnorm_args = {'track_running_stats':True,
                      'momentum':0.999,
                      'eps':1e-4,}

    coupling_args = {
            'subnet_constructor': None,
            'clamp': clamp,
            'act_norm': float(classifier.args['model']['act_norm']),
            'gin_block': False,
            'permute_soft': True,
        }

    def weights_init(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)
        if type(m) == nn.BatchNorm2d:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)
            m.weight.data *= 0.1

    def basic_residual_block(width, groups, dropout, relu_first, cin, cout):
        width = width * groups
        layers = []
        if relu_first:
            layers = [nn.ReLU()]
        else:
            layers = []

        layers.extend([
            nn.Conv2d(cin, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width, **batchnorm_args),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(width, **batchnorm_args),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),

            nn.Conv2d(width, cout, 1, padding=0)
        ])

        layers = nn.Sequential(*layers)
        layers.apply(weights_init)

        return layers

    def strided_residual_block(width, groups, cin, cout):
        width = width * groups
        layers = nn.Sequential(
            nn.Conv2d(cin, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width, **batchnorm_args),
            nn.ReLU(),

            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(width, **batchnorm_args),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, cout, 1, padding=0)
        )

        layers.apply(weights_init)

        return layers

    def fc_constr(c_in, c_out):
        net = [nn.Linear(c_in, fc_width),
               nn.ReLU(),
               nn.Dropout(p=dropouts_fc),
               nn.Linear(fc_width,  c_out)]

        net = nn.Sequential(*net)
        net.apply(weights_init)
        return net

    nodes = [Ff.InputNode(*ndim_input, name='input')]
    channels = classifier.input_channels

    if classifier.dataset == 'MNIST':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.Reshape, {'target_dim':(1, *classifier.dims)}))
        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {'rebalance':1.}))
        channels *= 4

    for i, (width, n_blocks) in enumerate(zip(conv_widths, n_coupling_blocks_conv)):
        if classifier.dataset == 'MNIST' and i==0:
            continue

        drop = dropouts[i]
        conv_constr = partial(basic_residual_block, width, groups, drop, True)
        conv_strided = partial(strided_residual_block, width*2, groups)
        conv_lowres = partial(basic_residual_block, width*2, groups, drop, False)

        if i == 0:
            conv_first = partial(basic_residual_block, width, groups, drop, False)
        else:
            conv_first = conv_constr

        nodes.append(Ff.Node(nodes[-1], AIO_Block, dict(coupling_args, subnet_constructor=conv_first), name=f'CONV_{i}_0'))
        for k in range(1, n_blocks):
            nodes.append(Ff.Node(nodes[-1], AIO_Block, dict(coupling_args, subnet_constructor=conv_constr), name=f'CONV_{i}_{k}'))

        if i < len(conv_widths) - 1:
            nodes.append(Ff.Node(nodes[-1], DownsampleCouplingBlock, {'subnet_constructor_low_res':conv_lowres,
                                                                      'subnet_constructor_strided':conv_strided,
                                                                      'clamp':clamp}, name=f'DOWN_{i}'))
            channels *= 4

    if use_dct:
        nodes.append(Ff.Node(nodes[-1].out0, DCTPooling2d, {'rebalance':0.5}, name='DCT'))
    else:
        nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {}, name='Flatten'))

    for k in range(n_coupling_blocks_fc):
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}, name=f'PERM_FC_{k}'))
        nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock, {'subnet_constructor':fc_constr, 'clamp':2.0}, name=f'FC_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes, verbose=verbose)
