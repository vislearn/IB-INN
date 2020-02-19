from os.path import join

from functools import partial
import pdb

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from numbers import Number

from torch.autograd import Variable


def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor


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

        try:
            return F.leaky_relu(skip + residual)
        except:
            pass

def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()


class ResnetClassifier(nn.Module):

    def __init__(self, args, classes=10):
        super().__init__()

        self.K = 128
        # currently replaced wioth skip_width

        #modules.append(nn.Linear(skip_width, 10))
        self.extract_features, skip_width = self.construct_resnet(args)
        self.encode = nn.Sequential(nn.Linear(48, self.K * 2))

        self.decode = nn.Sequential(nn.Linear(self.K, classes))

        #self.optimizer = torch.optim.SGD(self.optimizer_params, float(args['training']['lr']), momentum=float(args['training']['sgd_momentum']), weight_decay=1e-4)


    def construct_resnet(self, args, equivalent_channels=True):

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

        batchnorm_args = {'track_running_stats': True,
                          'momentum': 0.1,
                          'eps': 1e-4, }

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
                   nn.Linear(fc_width, c_out)]

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
                conv_strided = partial(strided_residual_block, conv_widths[i + 1], groups)
                conv_lowres = partial(basic_residual_block, conv_widths[i + 1], groups, drop, False)
                modules.append(ResNetBlock(skip_width, skip_widths[i + 1], conv_strided,
                                           extra_subnetwork=nn.Conv2d(skip_width, skip_widths[i + 1], 3, stride=2, padding=1)))

                modules.append(ResNetBlock(skip_widths[i + 1], skip_widths[i + 1], conv_lowres))

        modules.append(nn.AvgPool2d(32 // (2 ** (len(conv_widths) - 1))))
        modules.append(SqueezeFrom2d())

        for k in range(n_coupling_blocks_fc):
            modules.append(ResNetBlock(skip_width, skip_width, fc_constr))

        #modules.append(nn.Linear(skip_width, 10))

        return nn.Sequential(*modules), skip_width

    def forward(self, x, num_sample=1):
        features = self.extract_features(x)
        statistics = self.encode(features)

        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)

        logit = self.decode(encoding)

        if num_sample == 1:
            pass
        elif num_sample > 1:
            logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])

    def save(self, fname):
        state = self.state_dict()
        torch.save(state, fname)

class WrapperVIB(ResnetClassifier):
    def __init__(self, args):
        dataset = args['data']['dataset']
        if dataset == 'CIFAR100':
            n_classes = 100
        else:
            n_classes = 10
        super().__init__(args, classes=n_classes)
        self.args = args
        self.feed_forward = False

        self.trainable_params = list(self.parameters())
        self.trainable_params = list(filter(lambda p: p.requires_grad, self.trainable_params))

        self.dataset = self.args['data']['dataset']
        optimizer = self.args['training']['optimizer']
        base_lr = float(self.args['training']['lr'])
        optimizer_params = [ {'params':list(filter(lambda p: p.requires_grad, self.parameters()))},]

        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(optimizer_params, base_lr,
                                              momentum=float(self.args['training']['sgd_momentum']),
                                              weight_decay=float(self.args['training']['weight_decay']))
        elif optimizer == 'ADAM':
            self.optimizer = torch.optim.Adam(optimizer_params, base_lr,
                                              betas=eval(self.args['training']['adam_betas']),
                                              weight_decay=float(self.args['training']['weight_decay']))



    def load(self, fname):
        data = torch.load(fname)
        self.load_state_dict(data)

    def encoder(self, x):
        return super().forward(x)

    def forward(self, x, y=None, loss_mean=True, z_samples=1):
        (mu, std), logit = super().forward(x, num_sample=z_samples)

        info_loss = -0.5*(1 + 2*std.log() - mu.pow(2) - std.pow(2)).sum(1).div(np.log(2))
        info_loss /= 3072

        losses = {'logits_tr': logit,
                  'nll_joint_tr': info_loss,
                  'nll_class_tr': 0. * info_loss.detach()}

        if y is not None:
            class_loss = - (torch.log_softmax(logit, dim=1) * y).sum(1) / np.log(2.)
            acc = torch.mean((torch.max(y, dim=1)[1]
                           == torch.max(logit.detach(), dim=1)[1]).float())
            losses['cat_ce_tr'] = class_loss
            losses['acc_tr'] = acc

        if loss_mean:
            for k,v in losses.items():
                losses[k] = torch.mean(v)

        return losses

    def validate(self, x, y, eval_mode=True):
        is_train = self.training
        if eval_mode:
            self.eval()

        with torch.no_grad():
            losses = self.forward(x, y, loss_mean=False, z_samples=12)
            nll_joint, nll_class, cat_ce, logits, acc = (losses['nll_joint_tr'].mean(),
                                                         losses['nll_class_tr'].mean(),
                                                         losses['cat_ce_tr'].mean(),
                                                         losses['logits_tr'],
                                                         losses['acc_tr'])

            mu_dist = torch.Tensor((0.,)).cuda()

        if is_train:
            self.train()

        return {'nll_joint_val': nll_joint,
                'nll_class_val': nll_class,
                'logits_val':    logits,
                'cat_ce_val':    cat_ce,
                'acc_val':       acc,
                'delta_mu_val':  mu_dist}

args = {
    'ablations': {   'class_nll': 'False',
                     'feed_forward_resnet': 'False',
                     'no_nll_term': 'False',
                     'standard_softmax_loss': 'False'},
    'checkpoints': {   'base_name': 'default',
                       'checkpoint_when_crash': 'False',
                       'ensemble_index': 'None',
                       'global_output_folder': './output',
                       'interval_checkpoint': '1000',
                       'interval_figure': '200',
                       'interval_log': '200',
                       'live_updates': 'False',
                       'output_dir': './output/vib',
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
    'training': {   'adam_betas': '[0.9, 0.99]',
                    'adam_betas_mu': '[0.95, 0.99]',
                    'adam_betas_phi': '[0.95, 0.99]',
                    'aggmo_betas': '[0.0, 0.9, 0.99]',
                    'aggmo_betas_mu': '[0.0, 0.9, 0.99]',
                    'aggmo_betas_phi': '[0.0, 0.9, 0.99]',
                    'beta_ib': '1e-6',
                    'clip_grad_norm': '8.',
                    'empirical_mu': 'False',
                    'exponential_scheduler': 'False',
                    'lr': '0.007',
                    'lr_burn_in': '5',
                    'lr_mu': '0.4',
                    'lr_phi': '1.',
                    'n_epochs': '450',
                    'optimizer': 'SGD',
                    'scheduler_milestones': '[150, 250, 350]',
                    'sgd_momentum': '0.9',
                    'sgd_momentum_mu': '0.0',
                    'sgd_momentum_phi': '0.8',
                    'train_mu': 'True',
                    'train_phi': 'False',
                    'weight_decay': '1e-4'}}


if __name__ == '__main__':
    import math
    import os
    from data import Dataset

    dataset = Dataset(args)
    resnet = ResnetClassifier(args)
    resnet.cuda()
    resnet.weight_init()
    resnet.train()

    history = dict()
    history['avg_acc'] = 0.
    history['info_loss'] = 0.
    history['class_loss'] = 0.
    history['total_loss'] = 0.
    history['epoch'] = 0
    history['iter'] = 0

    #args['training']['n_epochs'] = 10
    args['training']['beta_ib'] = 1.

    parameters = list(filter(lambda p: p.requires_grad, resnet.parameters()))
    #optimizer = torch.optim.Adam(parameters, 1e-3, betas=(0.9, 0.999))
    optimizer = torch.optim.SGD(parameters, 0.07, momentum=0.9)
    sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1,
                                             milestones=eval(args['training']['scheduler_milestones']))

    N_epochs = int(args['training']['n_epochs'])
    print('total epochs are', N_epochs)

    beta = float(args['training']['beta_ib'])
    print('IB-INN-beta is %.4e' % (beta))
    beta = 1. / (beta * 3072)
    print('VIB-beta is %.4e' % (beta))

    interval_log = int(args['checkpoints']['interval_log'])
    interval_checkpoint = int(args['checkpoints']['interval_checkpoint'])
    interval_figure = int(args['checkpoints']['interval_figure'])
    save_on_crash = eval(args['checkpoints']['checkpoint_when_crash'])

    output_dir = args['checkpoints']['output_dir']
    resume = args['checkpoints']['resume_checkpoint']
    grad_clip = float(args['training']['clip_grad_norm'])
    label_smoothing = float(args['data']['label_smoothing'])
    os.makedirs(output_dir, exist_ok=True)

    plot_columns = ['XZ', 'XY', 'acc']
    global_iter = 0
    from time import time
    t_start = time()

    try:
        for i_epoch in range(N_epochs):
            for i_batch, (x,l) in tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader), disable=True):

                x, y = x.cuda(), dataset.onehot(l.cuda(), label_smoothing)
                (mu, std), logit = resnet(x)

                #class_loss = F.cross_entropy(logit, y).div(np.log(2))
                class_loss = - (torch.log_softmax(logit, dim=1) * y).sum(1).mean() / np.log(2.)
                info_loss = -0.5*(1 + 2*std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(np.log(2))
                total_loss = class_loss + beta * info_loss
                #print('\r %-15.3f   %-15.3f   %-15.3f' %(class_loss.item(), info_loss.item(), total_loss.item()), end='')
                if global_iter < 200:
                    total_loss *= 0.05

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
                optimizer.step()

                global_iter = global_iter +  1

            #print()
            sched.step()
            val_avg_losses = {}
            for l_name in plot_columns:
                val_avg_losses[l_name] = []

            resnet.eval()

            for val_batch, (x, y) in enumerate([(dataset.val_x, dataset.val_y)]):
                with torch.no_grad():
                    x = x.cuda()
                    y = y.cuda()

                    (mu, std), logit = resnet(x)

                    #class_loss = F.cross_entropy(logit, y).div(np.log(2))
                    class_loss = - (torch.log_softmax(logit, dim=1) * y).sum(1).mean() / np.log(2.)
                    info_loss = beta * (-0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean().div(np.log(2)))

                    prediction = F.softmax(logit, dim=1).max(1)[1]
                    gt = y.max(1)[1]
                    accuracy = torch.eq(prediction, gt).float().mean()

                    val_avg_losses['XZ'].append(np.mean(info_loss.item()))
                    val_avg_losses['XY'].append(np.mean(class_loss.item()))
                    val_avg_losses['acc'].append(np.mean(accuracy.item()))

            for l_name in plot_columns:
                val_avg_losses[l_name] = np.mean(val_avg_losses[l_name])

            print('%.6i\t %-16s: %.4f' % (global_iter, 'time', (time() - t_start) / 60.))
            for l_name in plot_columns:
                print('%.6i\t %-16s: %.4f' % (global_iter, l_name, val_avg_losses[l_name]))
            print('---'*20)

            resnet.train()

            if i_epoch > 0 and (i_epoch % interval_checkpoint) == 0:
                resnet.save(join(output_dir, f'model_{i_epoch}.pt'))

    except:
        if save_on_crash:
            resnet.save(join(output_dir, f'model_ABORT{ensemble_str}.pt'))
        raise

    resnet.save(join(output_dir, f'model.pt'))

