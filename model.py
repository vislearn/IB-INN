from functools import partial
import pdb

import torch
import torch.nn as nn
import numpy as np

import inn_architecture
import feed_forward_architecture
import data

class GenerativeClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        init_latent_scale        = eval(self.args['model']['mu_init'])
        weight_init              = eval(self.args['model']['weight_init'])
        self.dataset             = self.args['data']['dataset']
        self.ch_pad              = eval(self.args['data']['pad_noise_channels'])
        self.feed_forward        = eval(self.args['ablations']['feed_forward_resnet'])
        self.feed_forward_revnet = eval(self.args['ablations']['feed_forward_irevnet'])

        if self.dataset == 'MNIST':
            self.dims  = (28, 28)
            self.input_channels = 1
            self.ndim_tot = int(np.prod(self.dims))
            self.n_classes = 10
        elif self.dataset in ['CIFAR10', 'CIFAR100']:
            self.dims  = (3 + self.ch_pad, 32, 32)
            self.input_channels = 3 + self.ch_pad
            self.ndim_tot = int(np.prod(self.dims))
            if self.dataset == 'CIFAR10':
                self.n_classes = 10
            else:
                self.n_classes = 100
        else:
            raise ValueError(f"what is this dataset, {args['data']['dataset']}?")

        if self.feed_forward_revnet:
            self.feed_forward = True
            self.inn = inn_architecture.construct_irevnet(self)
        elif self.feed_forward:
            self.inn = feed_forward_architecture.constuct_resnet(self.args)
        else:
            self.inn = inn_architecture.constuct_inn(self)

        mu_populate_dims = self.ndim_tot
        init_scale = init_latent_scale / np.sqrt(2 * mu_populate_dims // self.n_classes)
        self.mu = nn.Parameter(torch.zeros(1, self.n_classes, self.ndim_tot))
        self.mu_empirical = eval(self.args['training']['empirical_mu'])

        for k in range(mu_populate_dims // self.n_classes):
            self.mu.data[0, :, self.n_classes * k : self.n_classes * (k+1)] = init_scale * torch.eye(self.n_classes)

        self.phi = nn.Parameter(torch.zeros(self.n_classes))

        self.trainable_params = list(self.inn.parameters())
        self.trainable_params = list(filter(lambda p: p.requires_grad, self.trainable_params))

        self.train_mu  = eval(self.args['training']['train_mu'])
        self.train_phi = eval(self.args['training']['train_mu'])
        self.train_inn = True

        optimizer = self.args['training']['optimizer']

        for p in self.trainable_params:
            p.data *= weight_init

        self.trainable_params += [self.mu, self.phi]
        base_lr = float(self.args['training']['lr'])

        optimizer_params = [ {'params':list(filter(lambda p: p.requires_grad, self.inn.parameters()))},]

        if self.train_mu:
            optimizer_params.append({'params': [self.mu],
                                     'lr': base_lr * float(self.args['training']['lr_mu']),
                                     'weight_decay': 0.})
            if optimizer == 'SGD':
                optimizer_params[-1]['momentum'] = float(self.args['training']['sgd_momentum_mu'])
            if optimizer == 'ADAM':
                optimizer_params[-1]['betas'] = eval(self.args['training']['adam_betas_mu'])
            if optimizer == 'AGGMO':
                optimizer_params[-1]['betas'] = eval(self.args['training']['aggmo_betas_mu'])

        if self.train_phi:
            optimizer_params.append({'params': [self.phi],
                                     'lr': base_lr * float(self.args['training']['lr_phi']),
                                     'weight_decay': 0.})
            if optimizer == 'SGD':
                optimizer_params[-1]['momentum'] = float(self.args['training']['sgd_momentum_phi'])
            if optimizer == 'ADAM':
                optimizer_params[-1]['betas'] = eval(self.args['training']['adam_betas_phi'])
            if optimizer == 'AGGMO':
                optimizer_params[-1]['betas'] = eval(self.args['training']['aggmo_betas_phi'])

        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(optimizer_params, base_lr,
                                              momentum=float(self.args['training']['sgd_momentum']),
                                              weight_decay=float(self.args['training']['weight_decay']))
        elif optimizer == 'ADAM':
            self.optimizer = torch.optim.Adam(optimizer_params, base_lr,
                                              betas=eval(self.args['training']['adam_betas']),
                                              weight_decay=float(self.args['training']['weight_decay']))
        elif optimizer == 'AGGMO':
            import aggmo
            self.optimizer = aggmo.AggMo(optimizer_params, base_lr,
                                              betas=eval(self.args['training']['aggmo_betas']),
                                              weight_decay=float(self.args['training']['weight_decay']))
        else:
            raise ValueError(f'what is this optimizer, {optimizer}?')

    def cluster_distances(self, z, y=None):

        if y is not None:
            mu = torch.mm(z.t().detach(), y.round())
            mu = mu / torch.sum(y, dim=0, keepdim=True)
            mu = mu.t().view(1, self.n_classes, -1)
            mu = 0.005 * mu + 0.995 * self.mu.data
            self.mu.data = mu.data

        z_i_z_i = torch.sum(z**2, dim=1, keepdim=True)   # batchsize x n_classes
        mu_j_mu_j = torch.sum(self.mu**2, dim=2)         # 1 x n_classes
        z_i_mu_j = torch.mm(z, self.mu.squeeze().t())    # batchsize x n_classes

        return -2 * z_i_mu_j + z_i_z_i + mu_j_mu_j

    def mu_pairwise_dist(self):

        mu_i_mu_j = self.mu.squeeze().mm(self.mu.squeeze().t())
        mu_i_mu_i = torch.sum(self.mu.squeeze()**2, 1, keepdim=True).expand(self.n_classes, self.n_classes)

        dist =  mu_i_mu_i + mu_i_mu_i.t() - 2 * mu_i_mu_j
        return torch.masked_select(dist, (1 - torch.eye(self.n_classes).cuda()).bool()).clamp(min=0.)

    def forward(self, x, y=None, loss_mean=True):

        if self.feed_forward:
            return self.losses_feed_forward(x, y, loss_mean)

        z = self.inn(x)
        jac = self.inn.log_jacobian(run_forward=False)

        log_wy = torch.log_softmax(self.phi, dim=0).view(1, -1)

        if self.mu_empirical and y is not None and self.inn.training:
            zz = self.cluster_distances(z, y)
        else:
            zz = self.cluster_distances(z)

        losses = {'L_x_tr':    (- torch.logsumexp(- 0.5 * zz + log_wy, dim=1) - jac ) / self.ndim_tot,
                  'logits_tr': - 0.5 * zz}

        log_wy = log_wy.detach()
        if y is not None:
            losses['L_cNLL_tr'] = (0.5 * torch.sum(zz * y.round(), dim=1) - jac) / self.ndim_tot
            losses['L_y_tr'] = torch.sum((torch.log_softmax(- 0.5 * zz + log_wy, dim=1) - log_wy) * y, dim=1)
            losses['acc_tr'] = torch.mean((torch.max(y, dim=1)[1]
                                        == torch.max(losses['logits_tr'].detach(), dim=1)[1]).float())

        if loss_mean:
            for k,v in losses.items():
                losses[k] = torch.mean(v)

        return losses

    def losses_feed_forward(self, x, y=None, loss_mean=True):
        logits = self.inn(x)

        losses = {'logits_tr': logits,
                  'L_x_tr': torch.zeros_like(logits[:,0])}

        if y is not None:
            ly =  torch.sum(torch.log_softmax(logits, dim=1) * y, dim=1)
            acc = torch.mean((torch.max(y, dim=1)[1]
                           == torch.max(logits.detach(), dim=1)[1]).float())
            losses['L_y_tr'] = ly
            losses['acc_tr'] = acc
            losses['L_cNLL_tr'] = torch.zeros_like(ly)

        if loss_mean:
            for k,v in losses.items():
                losses[k] = torch.mean(v)

        return losses

    def validate(self, x, y, eval_mode=True):
        is_train = self.inn.training
        if eval_mode:
            self.inn.eval()

        with torch.no_grad():
            losses = self.forward(x, y, loss_mean=False)
            l_x, class_nll, l_y, logits, acc = (losses['L_x_tr'].mean(),
                                                losses['L_cNLL_tr'].mean(),
                                                losses['L_y_tr'].mean(),
                                                losses['logits_tr'],
                                                losses['acc_tr'])

            mu_dist = torch.mean(torch.sqrt(self.mu_pairwise_dist()))

        if is_train:
            self.inn.train()

        return {'L_x_val':      l_x,
                'L_cNLL_val':   class_nll,
                'logits_val':   logits,
                'L_y_val':      l_y,
                'acc_val':      acc,
                'delta_mu_val': mu_dist}

    def reset_mu(self, dataset):
        mu = torch.zeros(1, self.n_classes, self.ndim_tot).cuda()
        counter = 0

        with torch.no_grad():
            for x, l in dataset.train_loader:
                x, y = x.cuda(), dataset.onehot(l.cuda(), 0.05)
                z = self.inn(x)
                mu_batch = torch.mm(z.t().detach(), y.round())
                mu_batch = mu_batch / torch.sum(y, dim=0, keepdim=True)
                mu_batch = mu_batch.t().view(1, self.n_classes, -1)

                mu += mu_batch
                counter += 1

            mu /= counter
        self.mu.data  = mu.data

    def sample(self, y, temperature=1.):
        z = temperature * torch.randn(y.shape[0], self.ndim_tot).cuda()
        mu = torch.sum(y.round().view(-1, self.n_classes, 1) * self.mu, dim=1)
        return self.inn(z, rev=True)

    def save(self, fname):
        torch.save({'inn': self.inn.state_dict(),
                    'mu':  self.mu,
                    'phi': self.phi,
                    'opt': self.optimizer.state_dict()}, fname)

    def load(self, fname):
        data = torch.load(fname)
        data['inn'] = {k:v for k,v in data['inn'].items() if 'tmp_var' not in k}
        self.inn.load_state_dict(data['inn'])
        self.mu.data.copy_(data['mu'].data)
        self.phi.data.copy_(data['phi'].data)
        try:
            pass
        except:
            print('loading the optimizer went wrong, skipping')
