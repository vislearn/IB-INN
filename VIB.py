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
        self.feed_forward = True
        self.feed_forward_revnet = False

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
                  'L_x_tr': info_loss,
                  'L_cNLL_tr': 0. * info_loss.detach()}

        if y is not None:
            class_loss = - (torch.log_softmax(logit, dim=1) * y).sum(1) / np.log(2.)
            acc = torch.mean((torch.max(y, dim=1)[1]
                           == torch.max(logit.detach(), dim=1)[1]).float())
            losses['L_y_tr'] = -class_loss
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
            info_loss, class_nll, l_y, logits, acc = (losses['L_x_tr'].mean(),
                                                      losses['L_cNLL_tr'].mean(),
                                                      losses['L_y_tr'].mean(),
                                                      losses['logits_tr'],
                                                      losses['acc_tr'])

            mu_dist = torch.Tensor((0.,)).cuda()

        if is_train:
            self.train()

        return {'L_x_val':      info_loss,
                'L_cNLL_val':   class_nll,
                'logits_val':   logits,
                'L_y_val':      l_y,
                'acc_val':      acc,
                'delta_mu_val': mu_dist}
