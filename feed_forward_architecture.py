from functools import partial
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SqueezeFrom2d(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class ResNetBlock(nn.Module):
    def __init__(self, ch_in, ch_out, subnetwork, extra_subnetwork=(lambda x: x)):
        super().__init__()
        self.net = subnetwork(ch_in, ch_out)
        self.extra = extra_subnetwork

    def forward(self, x):
        skip = self.extra(x)
        residual = self.net(x)
        return F.leaky_relu(skip + residual)

def constuct_resnet(args, equivalent_channels=True):

    fc_width = int(args['model']['fc_width'])
    n_coupling_blocks_fc = int(args['model']['n_coupling_blocks_fc'])

    conv_widths = eval(args['model']['conv_widths'])
    if equivalent_channels:
        skip_widths = [3, 12, 48]
    else:
        skip_widhts = conv_widths

    n_coupling_blocks_conv = eval(args['model']['n_coupling_blocks_conv'])
    dropouts = eval(args['model']['dropout_conv'])
    dropouts_fc = float(args['model']['dropout_fc'])

    groups = int(args['model']['n_groups'])
    clamp = float(args['model']['clamp'])

    ndim_input = (32, 32, 3)

    batchnorm_args = {'track_running_stats':True,
                      'momentum':0.1,
                      'eps':1e-4,}

    def weights_init(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)
        if type(m) == nn.BatchNorm2d:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

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


    modules = [nn.Conv2d(3, skip_widths[0], 1)]

    for i, (conv_width, skip_width, n_blocks) in enumerate(zip(conv_widths, skip_widths, n_coupling_blocks_conv)):
        drop = dropouts[i]

        conv_constr = partial(basic_residual_block, conv_width, groups, drop, True)

        if i == 0:
            conv_first = partial(basic_residual_block, conv_width, groups, drop, False)
        else:
            conv_first = conv_constr

        modules.append(ResNetBlock(skip_width, skip_width, conv_first))
        for k in range(1, n_blocks):
            modules.append(ResNetBlock(skip_width, skip_width, conv_constr))

        if i < len(conv_widths) - 1:
            conv_strided = partial(strided_residual_block, conv_widths[i+1], groups)
            conv_lowres = partial(basic_residual_block, conv_widths[i+1], groups, drop, False)
            modules.append(ResNetBlock(skip_width, skip_widths[i+1], conv_strided,
                                       extra_subnetwork=nn.Conv2d(skip_width, skip_widths[i+1], 3, stride=2, padding=1)))

            modules.append(ResNetBlock(skip_widths[i+1], skip_widths[i+1], conv_lowres))


    modules.append(nn.AvgPool2d(32 // (2**(len(conv_widths) - 1))))
    modules.append(SqueezeFrom2d())

    for k in range(n_coupling_blocks_fc):
        modules.append(ResNetBlock(skip_width, skip_width, fc_constr))

    if args['data']['dataset'] == 'CIFAR100':
        n_classes = 100
    else:
        n_classes = 10

    modules.append(nn.Linear(skip_width, n_classes))
    return nn.Sequential(*modules)


default_args = {
    'ablations': {   'class_nll': 'False',
                     'feed_forward_resnet': 'False',
                     'no_nll_term': 'False',
                     'standard_softmax_loss': 'False'},
    'checkpoints': {   'base_name': 'default',
                       'checkpoint_when_crash': 'False',
                       'ensemble_index': 'None',
                       'global_output_folder': '/scratch/ws/lyar092b-ibinn/',
                       'interval_checkpoint': '1000',
                       'interval_figure': '200',
                       'interval_log': '200',
                       'live_updates': 'False',
                       'output_dir': '/scratch/ws/lyar092b-ibinn/default',
                       'resume_checkpoint': ''},
    'data': {   'batch_size': '128',
                'dataset': 'CIFAR10',
                'label_smoothing': '0.07',
                'noise_amplitde': '0.015',
                'pad_noise_channels': '0',
                'pad_noise_std': '1.0',
                'tanh_augmentation': 'False'},
    'evaluation': {'ensemble_members': '1'},
    'model': {   'act_norm': '0.75',
                 'clamp': '0.7',
                 'conv_widths': '[16, 32, 64]',
                 'dropout_conv': '[0., 0., 0.25]',
                 'dropout_fc': '0.5',
                 'fc_width': '1024',
                 'mu_init': '5.0',
                 'n_coupling_blocks_conv': '[8, 24, 24]',
                 'n_coupling_blocks_fc': '1',
                 'n_groups': '1',
                 'weight_init': '1.0'},
    'training': {   'adam_betas': '[0.95, 0.99]',
                    'adam_betas_mu': '[0.95, 0.99]',
                    'adam_betas_phi': '[0.95, 0.99]',
                    'aggmo_betas': '[0.0, 0.9, 0.99]',
                    'aggmo_betas_mu': '[0.0, 0.9, 0.99]',
                    'aggmo_betas_phi': '[0.0, 0.9, 0.99]',
                    'beta_ib': '1.0',
                    'clip_grad_norm': '8.',
                    'empirical_mu': 'False',
                    'exponential_scheduler': 'False',
                    'lr': '0.07',
                    'lr_burn_in': '500',
                    'lr_mu': '0.4',
                    'lr_phi': '1.',
                    'n_epochs': '400',
                    'optimizer': 'SGD',
                    'scheduler_milestones': '[100, 200, 300]',
                    'sgd_momentum': '0.9',
                    'sgd_momentum_mu': '0.0',
                    'sgd_momentum_phi': '0.8',
                    'train_mu': 'True',
                    'train_phi': 'False',
                    'weight_decay': '1e-4'}}


if __name__ == '__main__':
    resnet = constuct_resnet(default_args)
    resnet.cuda()
    z = resnet(torch.randn(1, 3, 32, 32).cuda())
    print(resnet)
    print(z.shape)
