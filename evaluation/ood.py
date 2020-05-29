import os

import torch
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import GenerativeClassifier


def outlier_detection(inn_model, data, test_set=False):
    ''' the option `test_set` controls, whether the test set, or the validation set is used.'''

    import ood_datasets.imagenet
    import ood_datasets.cifar
    import ood_datasets.quickdraw
    import ood_datasets.svhn

    ensemble = int(inn_model.args['evaluation']['ensemble_members'])
    inn_ensemble = [inn_model]

    if ensemble > 1:
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

    scores_all = {}
    entrop_all = {}
    generators = []

    generators = [
                      (ood_datasets.cifar.cifar_rgb_rotation(inn_model.args, 0.35), 'rot_%.3f' % (0.3)),
                      (ood_datasets.quickdraw.quickdraw_colored(inn_model.args), 'Quickdraw'),
                      (ood_datasets.cifar.cifar_noise(inn_model.args, 0.01), 'Noisy'),
                      (ood_datasets.imagenet.imagenet(inn_model.args), 'ImageNet'),
                      (ood_datasets.svhn.svhn(inn_model.args), 'SVHN'),
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

