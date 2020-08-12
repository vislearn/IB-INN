from . import output
from .ood import outlier_detection
from .calibration import calibration_curve
from .latent_space import show_samples, show_latent_space
from .test_metrics import test_metrics

import os

import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from model import GenerativeClassifier
import data

def average_batch_norm(model, data, N_epochs=5):
    '''Make the batch norm statistics more accurate:
       iterate over the entire training set N_epochs times with fixed weights.
       Especially for small models, the running stats tend to not be so stable during training.'''

    # because of FrEIA, there are so many layers and layers of subnetworks...
    instance_counter = 0
    for node in model.inn.children():
        for module in node.children():
            for subnet in module.children():
                for layer in subnet.children():
                    if type(layer) == torch.nn.Sequential:
                        for l in layer.children():
                            if type(l) == torch.nn.BatchNorm2d:
                                l.reset_running_stats()
                                l.momentum = None
                                instance_counter += 1
                    if type(layer) == torch.nn.BatchNorm2d:
                        layer.reset_running_stats()
                        layer.momentum = None
                        instance_counter += 1

    assert instance_counter > 0, "No batch norm layers found. Is the model constructed differently?"
    model.train()
    progress_bar = tqdm(total=N_epochs * len(data.train_loader),
                        ncols=120, ascii=True, mininterval=1.)
    with torch.no_grad():
        for i in range(N_epochs):
            for x, l in data.train_loader:
                x = x.cuda()
                z = model.inn(x)
                progress_bar.update()

    progress_bar.close()
    print(f'\n>> Reset {instance_counter} instances of torch.nn.BatchNorm2d')
    model.eval()

def average_batch_norm_vib(model, data, N_epochs=5):
    '''Make the batch norm statistics more accurate:
       iterate over the entire training set N_epochs times with fixed weights.
       Especially for small models, the running stats tend to not be so stable during training.
       Same function for VIB and other feed-forward models.'''

    for module in model.children():
        if type(module) == torch.nn.BatchNorm2d:
            module.reset_running_stats()
            module.momentum = None
        for subnet in module.children():
            if type(subnet) == torch.nn.BatchNorm2d:
                subnet.reset_running_stats()
                subnet.momentum = None
            for layer in subnet.children():
                if type(layer) == torch.nn.BatchNorm2d:
                    layer.reset_running_stats()
                    layer.momentum = None
    model.train()
    progress_bar = tqdm(total=N_epochs * len(data.train_loader),
                        ncols=120, ascii=True, mininterval=1.)
    with torch.no_grad():
        for i in range(N_epochs):
            for x, l in data.train_loader:
                x = x.cuda()
                losses = model(x)
                progress_bar.update()

    progress_bar.close()
    model.eval()


def val_plots(fname, model, data):
    n_classes = data.n_classes
    n_samples = 4
    n_classes_show = min(10, n_classes)

    y_digits = torch.zeros(n_classes_show * n_samples, n_classes).cuda()
    for i in range(n_classes_show):
        y_digits[n_samples * i : n_samples * (i+1), i] = 1.

    show_samples(model, data, y_digits)
    show_latent_space(model, data)

    with PdfPages(fname) as pp:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')

    plt.close('all')

def test(args):

    output_dir = args['checkpoints']['output_dir']
    model_fname = os.path.join(output_dir, 'model.pt')
    fig_fname = os.path.join(output_dir, 'figs.pdf')

    do_ood = eval(args['evaluation']['ood'])
    vib_model = eval(args['ablations']['vib'])


    print('>> Plotting loss curves')
    try:
        losses = np.loadtxt(os.path.join(output_dir, 'losses.dat'),
                        usecols = [0] + list(range(3,10)),
                        skiprows = 1).T
    except OSError:
        try:
            losses = np.loadtxt(os.path.join(output_dir, 'losses.00.dat'),
                        usecols = [0] + list(range(3,10)),
                        skiprows = 1).T
        except OSError:
            print('>> Skipping Loss Curves')
            losses = None

    if losses is not None:
        plt.figure(figsize=(8,12))
        plt.subplot(2,1,1)
        plt.plot(losses[0], losses[1], '--', color='red', label='$\mathcal{L}_X$ (train)')
        plt.plot(losses[0], losses[3], '--', color='blue', label='$\mathcal{L}_Y$ (train)')
        plt.plot(losses[0], losses[2], color='red', label='$\mathcal{L}_X$ (val)')
        plt.plot(losses[0], losses[4], color='blue', label='$\mathcal{L}_X$ (val)')
        plt.grid(True)
        plt.ylim(-5, 0)
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(losses[0], 100 * (1. - losses[5]), '--', color='orange', label='err (val)')
        plt.plot(losses[0], 100 * (1. - losses[6]), color='orange', label='err (train)')
        plt.ylim([5, 50])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    print('>> Loading dataset')
    dataset = data.Dataset(args)
    n_classes = dataset.n_classes
    n_samples = 10

    print('>> Constructing model')
    if vib_model:
        from VIB import WrapperVIB
        inn = WrapperVIB(args)
    else:
        inn = GenerativeClassifier(args)
    inn.cuda()

    print('>> Loading weights')
    # first, try to load the model with averaged batch norms.
    # if not, average out the batch norms and re-save it.
    try:
        try:
            inn.load(model_fname[:-3] + '.avg.pt')
        except FileNotFoundError:
            # use the first ensemble member if this is an ensemble model
            inn.load(model_fname[:-3] + '.00.avg.pt')
    except FileNotFoundError:
        try:
            inn.load(model_fname)
        except FileNotFoundError:
            # use the first ensemble member if this is an ensemble model
            inn.load(model_fname[:-3] + '.00.pt')

        print('>> Averaging BatchNorm')
        if vib_model:
            average_batch_norm_vib(inn, dataset, int(args['evaluation']['train_set_oversampling']))
        else:
            average_batch_norm(inn, dataset, int(args['evaluation']['train_set_oversampling']))
        inn.eval()

        try:
            for k in list(inn.inn._buffers.keys()):
                if 'tmp_var' in k:
                    del inn.inn._buffers[k]
        except AttributeError:
            # Feed-forward nets dont have the wierd FrEIA problems, skip
            pass

        inn.save(model_fname[:-3] + '.avg.pt')
    inn.eval()

    print('>> Determining test accuracy')
    metrics = test_metrics(inn, dataset, args)
    # the numbers are np.float32, and json won't take it if not cast to float() explicitly.
    results_dict = {'test_metrics': metrics}

    print('>> Plotting calibration curve')
    ece, mce, ice, ovc = calibration_curve(inn, dataset)
    results_dict['calib_err'] = {'ece': float(100. * ece),
                                 'mce': float(100. * mce),
                                 'ice': float(100. * ice),
                                 'oce': float(ovc),
                                 'gme': float(100. * (ece*mce*ice)**0.333333333)}

    if not vib_model and not inn.feed_forward:
        print('>> Plotting generated samples')
        n_classes = dataset.n_classes
        y_digits = torch.zeros(10 * n_samples, n_classes).cuda()
        for i in range(10):
            y_digits[n_samples * i : n_samples * (i+1), i] = 1.
        show_samples(inn, dataset, y_digits)

        print('>> Plotting latent space')
        show_latent_space(inn, dataset, test_set=True)

    if do_ood:
        print('>> Determining outlier AUC')
        aucs_1t, aucs_2t, aucs_tt, entrop, delta_entrop = outlier_detection(inn, dataset, args, test_set=True)

        for aucs, test_type in zip([aucs_1t, aucs_2t, aucs_tt, entrop, delta_entrop],
                                   ['ood_1t', 'ood_2t', 'ood_tt', 'ood_ent', 'ood_d_ent']):
            m = len(list(aucs.keys()))
            geo_mean = np.prod(list(aucs.values())) ** (1./m)
            ari_mean = np.mean(list(aucs.values()))

            aucs['geo_mean'] = geo_mean
            aucs['ari_mean'] = ari_mean

            for k,v in aucs.items():
                aucs[k] = float(v)

            results_dict[test_type] = aucs

    print('>> Saving figures')
    with PdfPages(fig_fname) as pp:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')

    print('>> Generating data output files')
    output.to_json(results_dict, output_dir)
    output.to_console(results_dict, output_dir)
    output.to_latex_table_row(results_dict, output_dir,
                              name=args['checkpoints']['base_name'],
                              italic_ood=False,
                              blank_ood=(inn.feed_forward or inn.feed_forward_revnet),
                              italic_entrop=False,
                              blank_bitspdim=(inn.feed_forward or inn.feed_forward_revnet),
                              blank_classif=(eval(args['training']['beta_IB']) == 0))
