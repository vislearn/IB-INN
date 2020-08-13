from os.path import join
from time import time
import sys

import numpy as np
from tqdm import tqdm
import torch.optim

import data
from model import GenerativeClassifier
from VIB import WrapperVIB
import evaluation

def train(args):

    N_epochs            = eval(args['training']['n_epochs'])
    beta                = eval(args['training']['beta_IB'])
    train_nll           = bool(not eval(args['ablations']['no_NLL_term']))
    train_class_nll     = eval(args['ablations']['class_NLL'])
    label_smoothing     = eval(args['data']['label_smoothing'])
    grad_clip           = eval(args['training']['clip_grad_norm'])
    train_vib           = eval(args['ablations']['vib'])

    interval_log        = eval(args['checkpoints']['interval_log'])
    interval_checkpoint = eval(args['checkpoints']['interval_checkpoint'])
    interval_figure     = eval(args['checkpoints']['interval_figure'])
    save_on_crash       = eval(args['checkpoints']['checkpoint_when_crash'])

    output_dir          = args['checkpoints']['output_dir']
    resume              = args['checkpoints']['resume_checkpoint']
    ensemble_index      = eval(args['checkpoints']['ensemble_index'])

    if ensemble_index is None:
        ensemble_str = ''
    else:
        ensemble_str = '.%.2i' % (ensemble_index)
    logfile = open(join(output_dir, f'losses{ensemble_str}.dat'), 'w')
    live_loss = eval(args['checkpoints']['live_updates'])

    if train_vib:
        inn = WrapperVIB(args)
    else:
        inn = GenerativeClassifier(args)

    inn.cuda()
    dataset = data.Dataset(args)

    def log_write(line, endline='\n'):
        print(line, flush=True)
        logfile.write(line)
        logfile.write(endline)

    plot_columns = ['time', 'epoch', 'iteration',
                    'L_x_tr',
                    'L_x_val',
                    'L_y_tr',
                    'L_y_val',
                    'acc_tr',
                    'acc_val',
                    'delta_mu_val']

    train_loss_names = [l for l in plot_columns if l[-3:] == '_tr']
    val_loss_names   = [l for l in plot_columns if l[-4:] == '_val']

    header_fmt = '{:>15}' * len(plot_columns)
    output_fmt = '{:15.1f}      {:04d}/{:04d}      {:04d}/{:04d}' + '{:15.5f}' * (len(plot_columns) - 3)
    output_fmt_live = '{:15.1f}      {:04d}/{:04d}      {:04d}/{:04d}'
    for l_name in plot_columns[3:]:
        if l_name in train_loss_names:
            output_fmt_live += '{:15.5f}'
        else:
            output_fmt_live += '{:>15}'.format('')

    if eval(args['training']['exponential_scheduler']):
        print('Using exponential scheduler')
        sched = torch.optim.lr_scheduler.StepLR(inn.optimizer, gamma=0.002 ** (1/N_epochs), step_size=1)
    else:
        print('Using milestone scheduler')
        sched = torch.optim.lr_scheduler.MultiStepLR(inn.optimizer, gamma=0.1,
                                                 milestones=eval(args['training']['scheduler_milestones']))

    log_write(header_fmt.format(*plot_columns))

    if resume:
        inn.load(resume)

    t_start = time()
    if train_nll:
        beta_x = 2. / (1 + beta)
        beta_y = 2. * beta / (1 + beta)
    else:
        beta_x, beta_y = 0., 1.

    try:
        for i_epoch in range(N_epochs):
            running_avg = {l: [] for l in train_loss_names}

            for i_batch, (x,l) in enumerate(dataset.train_loader):

                x, y = x.cuda(), dataset.onehot(l.cuda(), label_smoothing)
                losses = inn(x, y)

                if train_class_nll:
                    loss = 2. * losses['L_cNLL_tr']
                else:
                    loss = beta_x * losses['L_x_tr'] - beta_y * losses['L_y_tr']
                loss.backward()

                torch.nn.utils.clip_grad_norm_(inn.trainable_params, grad_clip)
                inn.optimizer.step()
                inn.optimizer.zero_grad()

                if live_loss:
                    print(output_fmt_live.format(*([(time() - t_start) / 60.,
                                              i_epoch, N_epochs,
                                              i_batch, len(dataset.train_loader)]
                                             + [losses[l].item() for l in train_loss_names])),
                      flush=True, end='\r')

                for l_name in train_loss_names:
                    running_avg[l_name].append(losses[l_name].item())

                if not i_batch % interval_log:
                    for l_name in train_loss_names:
                        running_avg[l_name] = np.mean(running_avg[l_name])

                    val_losses = inn.validate(dataset.val_x, dataset.val_y)
                    for l_name in val_loss_names:
                        running_avg[l_name] = val_losses[l_name].item()

                    losses_display = [(time() - t_start) / 60.,
                                      i_epoch, N_epochs,
                                      i_batch, len(dataset.train_loader)]

                    losses_display += [running_avg[l] for l in plot_columns[3:]]
                    #TODO visdom?
                    log_write(output_fmt.format(*losses_display))
                    running_avg = {l: [] for l in train_loss_names}

            sched.step()
            if i_epoch > 2 and (val_losses['L_x_val'].item() > 1e5 or not np.isfinite(val_losses['L_x_val'].item())):
                if high_loss:
                    raise RuntimeError("loss is astronomical")
                else:
                    high_loss = True
            else:
                high_loss = False

            if i_epoch > 0 and (i_epoch % interval_checkpoint) == 0:
                inn.save(join(output_dir, f'model_{i_epoch}{ensemble_str}.pt'))
            if (i_epoch % interval_figure) == 0 and not inn.feed_forward and not train_vib:
                evaluation.val_plots(join(output_dir, f'figs_{i_epoch}{ensemble_str}.pdf'), inn, dataset)
    except:
        if save_on_crash:
            inn.save(join(output_dir, f'model_ABORT{ensemble_str}.pt'))
        raise
    finally:
        logfile.close()

    try:
        for k in list(inn.inn._buffers.keys()):
            if 'tmp_var' in k:
                del inn.inn._buffers[k]
    except AttributeError:
        # Feed-forward nets dont have the wierd FrEIA problems, skip
        pass

    inn.save(join(output_dir, f'model{ensemble_str}.pt'))
