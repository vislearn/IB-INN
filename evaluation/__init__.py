import glob
import os

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages

from model import GenerativeClassifier
import data

def average_batch_norm_vib(model, data):
    for module in model.children():
        if type(module) == torch.nn.BatchNorm2d:
            print(1)
            module.reset_running_stats()
            module.momentum = None
        for subnet in module.children():
            if type(subnet) == torch.nn.BatchNorm2d:
                print(2)
                subnet.reset_running_stats()
                subnet.momentum = None
            for layer in subnet.children():
                if type(layer) == torch.nn.BatchNorm2d:
                    print(3)
                    layer.reset_running_stats()
                    layer.momentum = None
    model.train()
    with torch.no_grad():
        count = 0
        for x, l in data.train_loader:
            x = x.cuda()
            losses = model(x)
            count += 1
            #if count > 100:
                #break

    model.eval()

def average_batch_norm(model, data):
    # because of FrEIA, there are so many layers and layers of subnetworks...
    for node in model.inn.children():
        for module in node.children():
            for subnet in module.children():
                for layer in subnet.children():
                    if type(layer) == torch.nn.BatchNorm2d:
                        layer.reset_running_stats()
                        layer.momentum = None

    model.train()
    with torch.no_grad():
        count = 0
        for x, l in data.train_loader:
            x = x.cuda()
            z = model.inn(x)
            count += 1
            #if count > 100:
                #break

    model.eval()

def show_samples(model, data, y, T=0.75):
    with torch.no_grad():
        samples = model.sample(y, T)
        samples = data.de_augment(samples).cpu().numpy()
        samples = np.clip(samples, 0, 1)

    w = min(y.shape[1], 10)
    h = int(np.ceil(y.shape[0] / w))

    plt.figure()
    for k in range(y.shape[0]):
        plt.subplot(h, w, k+1)
        if data.dataset == 'MNIST':
            plt.imshow(samples[k], cmap='gray')
        else:
            plt.imshow(samples[k].transpose(1,2,0))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()

def show_latent_space(model, data, test_set=False):
    ''' the option `test_set` controls, whether the test set, or the validation set is used.'''

    clusters = model.mu.data.cpu().numpy().squeeze()
    pca = PCA(n_components=2)
    pca.fit(clusters)

    mu_red = pca.transform(clusters)
    z_red = []
    true_label = []

    data_generator = (data.test_loader if test_set else [(data.val_x, torch.argmax(data.val_y, dim=1))])

    with torch.no_grad():
        for x, y in data_generator:
            true_label.append(y.cpu().numpy())
            x, y = x.cuda(), data.onehot(y.cuda())
            if isinstance(model, GenerativeClassifier):
                z = model.inn(x).cpu().numpy()
            else:
                (z, sig), logits = model.encoder(x)
                z = z.cpu().numpy()
            z_red.append(pca.transform(z))

    z_red = np.concatenate(z_red, axis=0)
    true_label= np.concatenate(true_label, axis=0)

    plt.figure()
    plt.scatter(mu_red[:,0], mu_red[:,1], c=np.arange(data.n_classes), cmap='tab10', s=250, alpha=0.5)
    plt.scatter(z_red[:,0], z_red[:,1], c=true_label, cmap='tab10', s=1)
    plt.tight_layout()

def outlier_detection(inn_model, data, test_set=False):
    ''' the option `test_set` controls, whether the test set, or the validation set is used.'''

    import ood_datasets.imagenet
    import ood_datasets.cifar
    import ood_datasets.quickdraw
    import ood_datasets.svhn

    ensemble = int(inn_model.args['evaluation']['ensemble_members'])
    inn_ensemble = [inn_model]
    if ensemble:
        print('>> Loading WAIC ensemble', end='')
        for i in range(1, ensemble):
            print('.', end='', flush=True)
            inn_ensemble.append(GenerativeClassifier(inn_model.args))
            model_fname = os.path.join(inn_model.args['checkpoints']['output_dir'], 'model.%.2i.pt' % (i))
            inn_ensemble[-1].load(model_fname)
            inn_ensemble[-1].cuda()
            inn_ensemble[-1].eval()

    def collect_scores(generator):
        with torch.no_grad():
            scores_cumul = []
            entrop_cumul = []
            print('.', end='', flush=True)
            for x in generator:
                x = x[0].cuda()
                ll_joint = []
                for inn in inn_ensemble:
                    losses = inn(x, y=None, loss_mean=False)
                    ll_joint.append(losses['nll_joint_tr'].cpu().numpy())
                entrop = torch.sum(- torch.softmax(losses['logits_tr'], dim=1)
                                   * torch.log_softmax(losses['logits_tr'], dim=1), dim=1).cpu().numpy()

                ll_joint = np.stack(ll_joint, axis=1)
                scores_cumul.append(np.mean(ll_joint, axis=1) + np.var(ll_joint, axis=1))
                entrop_cumul.append(entrop)
        return np.concatenate(scores_cumul), np.concatenate(entrop_cumul)

    in_distrib_data = (data.test_loader if test_set else [(data.val_x, torch.argmax(data.val_y, dim=1))])

    #from torchvision.datasets import FashionMNIST
    #from torch.utils.data import DataLoader
    #fashion_generator  = DataLoader(FashionMNIST('./fashion_mnist', download=True, train=False, transform=data.transform),
                                    #batch_size=data.batch_size, num_workers=8)


    scores_all = {}
    entrop_all = {}
    generators = []
    #for a in np.linspace(0., 1., 12):
        #generators.append((ood_datasets.cifar.cifar_rgb_rotation(inn_model.args, a), 'rot_%.3f' % (a)))

    generators = [
                      (ood_datasets.cifar.cifar_rgb_rotation(inn_model.args, 0.35), 'rot_%.3f' % (0.3)),
                      (ood_datasets.quickdraw.quickdraw_colored(inn_model.args), 'Qucikdraw'),
                      (ood_datasets.cifar.cifar_noise(inn_model.args, 0.01), 'Noisy'),
                      (ood_datasets.imagenet.imagenet(inn_model.args), 'ImageNet'),
                      #(ood_datasets.svhn.svhn(inn_model.args), 'SVHN'),
                      ]

    for gen,label in generators:
        scores_all[label], entrop_all[label] = collect_scores(gen)
        entrop_all[label] = np.mean(entrop_all[label])

    scores_ID, entrop_ID = collect_scores(in_distrib_data)

    fig_roc = plt.figure(figsize=(8,8))
    plt.plot([0,1], [0,1], '--', color='gray', label='random')
    plt.grid(True)

    fig_hist = plt.figure(figsize=(8,6))
    plt.hist(scores_ID, bins=50, histtype='step', density=True, color='gray', label='orig. distrib.')

    def auc(x1, x2, label=''):
        xjoint = -np.sort(-np.concatenate((x1, x2)))
        xjoint[-1] -= 0.0001
        val_range = (np.min(xjoint), np.max(xjoint))

        roc = []
        for x in xjoint:
            fpr = np.mean(x1 > x)
            tpr = np.mean(x2 > x)
            roc.append((fpr, tpr))
        roc = np.array(roc).T

        auc = np.trapz(roc[1], x=roc[0])
        plt.figure(fig_hist.number)
        plt.hist(x2, bins=50, histtype='step', density=True, label=label + ' (%.4f AUC)' % (auc))
        plt.figure(fig_roc.number)
        plt.plot(roc[0], roc[1], label=label + ' (%.4f AUC)' % (auc))

        return auc

    aucs = {}
    for label, score in scores_all.items():
        aucs[label] = auc(scores_ID, score, label)

    plt.figure(fig_hist.number)
    plt.legend()
    plt.figure(fig_roc.number)
    plt.legend()

    return aucs, entrop_all

def calibration_curve(model, data):

    pred = []
    gt = []
    correct_pred = []
    max_pred = []

    with torch.no_grad():
        for x, l in data.test_loader:
            x, y = x.cuda(), data.onehot(l.cuda())
            logits = model(x, y, loss_mean=False)['logits_tr']

            pred.append(torch.softmax(logits, dim=1).cpu().numpy())
            gt.append(y.cpu().numpy())
            correct_pred.append((torch.argmax(logits, dim=1) == l.cuda()).cpu().numpy())
            max_pred.append(torch.max(torch.softmax(logits, dim=1), dim=1)[0].cpu().numpy())

    pred = np.concatenate(pred, axis=0).flatten()
    gt   = np.concatenate(gt,   axis=0).astype(np.bool).flatten()
    correct_pred = np.concatenate(correct_pred, axis=0).astype(np.bool).flatten()
    max_pred = np.concatenate(max_pred, axis=0).flatten()

    if model.dataset == 'MNIST':
        points_in_bin = 200
    else:
        points_in_bin = 1000

    n_bins = np.ceil(pred.size / points_in_bin)
    #pred_bins = np.quantile(pred + 1e-5 * np.random.random(pred.shape),
                            #np.linspace(0., 1., n_bins))
    pred_bins = np.concatenate([np.linspace(-1e-6, 5e-2, 16),
                                np.linspace(5e-2, 1-5e-2, 22)[1:-1],
                                1. - np.linspace(5e-2, -1e-6, 16),])

    correct = pred[gt]
    wrong = pred[np.logical_not(gt)]

    hist_correct, _ = np.histogram(correct, bins=pred_bins)
    hist_wrong, _   = np.histogram(wrong,   bins=pred_bins)
    hist_tot = hist_correct + hist_wrong

    # only use bins with more than 10 samples, as it is too noisy below that
    bin_mask = (hist_tot > 20)
    hist_correct = hist_correct[bin_mask]
    hist_wrong = hist_wrong[bin_mask]
    hist_tot = hist_tot[bin_mask]

    q = hist_correct / hist_tot
    p = 0.5 * (pred_bins[1:] + pred_bins[:-1])
    p = p[bin_mask]

    poisson_err = q * np.sqrt(1 / (hist_correct + 1) + 1 / hist_tot)

    plt.figure(figsize=(5, 5))
    plt.errorbar(p, q, yerr=poisson_err, capsize=4, fmt='-o')
    plt.fill_between(p, q - poisson_err, q + poisson_err, alpha=0.25)
    plt.plot([0,1], [0,1], color='black')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Confidence')
    plt.ylabel('Fraction of correct pred.')
    plt.grid(True)
    plt.tight_layout()

    # Compute 'overconfidence'. might be better indicator than ECE, but not widely used
    overconfidence = []
    for confidence_thresh in 1e-2, 1e-3, 1e-4:
        confident = (max_pred > (1 - confidence_thresh))
        overconfidence.append(np.mean((1 - correct_pred)[confident]) /  confidence_thresh)

    # 'expected calibration error', see weinberger paper on calibration
    ece = np.sum(np.abs(q-p) * hist_tot) / np.sum(hist_tot)
    mce = np.max(np.abs(q-p))
    ice = np.trapz(np.abs(q-p), x=p)
    ovc = (1. - np.mean(q[-2:])) / (1. - pred_bins[-2])
    print('>> confidence thresshold', pred_bins[-2])
    return ece, mce, ice, ovc

def test_acc(inn, data):
    acc = []
    with torch.no_grad():
        for x, y in data.test_loader:
            x, y = x.cuda(), data.onehot(y.cuda())
            acc.append(inn.validate(x, y)['acc_val'].item())

    return np.mean(acc)

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

    model_fname = os.path.join(args['checkpoints']['output_dir'], 'model.pt')
    fig_fname = os.path.join(args['checkpoints']['output_dir'], 'figs.pdf')
    results_fname = os.path.join(args['checkpoints']['output_dir'], 'results.txt')
    do_ood = eval(args['evaluation']['ood'])
    vib_model = eval(args['ablations']['vib'])

    logfile = open(os.path.join(args['checkpoints']['output_dir'], f'results.dat'), 'w')

    def log_write(line, endline='\n'):
        print('\t' + line, flush=True)
        logfile.write(line)
        logfile.write(endline)

    print('>> Plotting loss curves')
    try:
        losses = np.loadtxt(os.path.join(args['checkpoints']['output_dir'], 'losses.dat'),
                        usecols = [0] + list(range(3,10)),
                        skiprows = 1).T
    except OSError:
        try:
            losses = np.loadtxt(os.path.join(args['checkpoints']['output_dir'], 'losses.00.dat'),
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
    try:
        inn.load(model_fname)
    except FileNotFoundError:
        # use the first ensemble member if this is an ensemble model
        inn.load(model_fname[:-3] + '.00.pt')

    print('>> Averaging BatchNorm')
    if vib_model:
        average_batch_norm_vib(inn, dataset)
    else:
        average_batch_norm(inn, dataset)
    inn.eval()

    print('>> Determining test accuracy')
    acc = test_acc(inn, dataset)
    log_write('ACC     %.4f' % (100 * acc))
    log_write('ACC ERR %.4f' % (100 - 100 * acc))
    log_write('LATEX %.2f &' % (100 - 100 * acc))

    print('>> Plotting calibration curve')
    ece, mce, ice, ovc = calibration_curve(inn, dataset)
    log_write(('XCE     ' + '%-10s' * 3) % ('ECE', 'MCE', 'OVC'))
    log_write(('XCE     ' + '%-10.6f' * 3) % (100. * ece, 100. * mce, ovc))
    log_write('XCE GM  %.6f' % (21.5443 * (ece*mce*max(1,ovc))**0.3333333333333))
    log_write('XCE GMO %.6f' % (21.5443 * (ece*mce*ovc)**0.3333333333333))
    log_write('LATEX %.3f & %.2f & %.2f & %.2f & ' % ((21.5443 * (ece*mce*max(1,ovc))**0.3333333333333),
                                                                100. * ece, 100. * mce, ovc))

    if not vib_model and not inn.feed_forward:
        print('>> Plotting generated samples')
        n_classes = dataset.n_classes
        y_digits = torch.zeros(10 * n_samples, n_classes).cuda()
        for i in range(10):
            y_digits[n_samples * i : n_samples * (i+1), i] = 1.
        show_samples(inn, dataset, y_digits)

        print('>> Plotting latent space')
        show_latent_space(inn, dataset, test_set=True)

    auc_records, ent_records = [], []
    if do_ood:
        print('>> Determining outlier AUC')
        aucs, entrop = outlier_detection(inn, dataset, test_set=True)
        print()
        for label, auc in aucs.items():
            printstr = 'AUC %-24s %.5f\n' % (label, aucs[label])
            auc_records.append(printstr)

        for label, ent in entrop.items():
            printstr = 'ENT %-24s %.5f\n' % (label, entrop[label])
            ent_records.append(printstr)

        labels_list = list(aucs.keys())
        log_write('DATASET ' + ''.join(['%-12s' % (l) for l in labels_list]))
        log_write('AUC     ' + ''.join(['%-12.4f' % (100 * aucs[l]) for l in labels_list]))
        log_write('AUC ERR ' + ''.join(['%-12.4f' % (100 - 100 * aucs[l]) for l in labels_list]))
        log_write('ENT     ' + ''.join(['%-12.4f' % (entrop[l]) for l in labels_list]))
        log_write('AUC GM  %.6f' % (100. * np.prod(np.array(list(aucs.values()))) ** (1./len(labels_list))))
        log_write('ENT GM  %.6f' % (np.prod(np.array(list(entrop.values()))) ** (1./len(labels_list))))

        log_write('LATEX   %.2f & ' % (100. * np.prod(np.array(list(aucs.values()))) ** (1./len(labels_list)))
                + ' & '.join(['%-8.1f' % (100 * aucs[l]) for l in labels_list]) + '&')
        log_write('LATEX   %.3f & ' % (np.prod(np.array(list(entrop.values()))) ** (1./len(labels_list)))
                + ' & '.join(['%-8.2f' % (entrop[l]) for l in labels_list]))

    print('>> Writing results to file')
    with open(results_fname, 'w') as f:
        f.write('acc %.6f\n' % (acc))
        f.write('ece %.6f\n' % (ece))
        f.write('mce %.6f\n' % (mce))
        f.write('ice %.6f\n' % (ice))
        f.write('\n')
        f.writelines(auc_records)
        f.writelines(ent_records)

    logfile.close()
    print('>> Saving figures')
    with PdfPages(fig_fname) as pp:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
