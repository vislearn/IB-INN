import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    pred_bins = np.concatenate([np.linspace(-1e-6, 5e-2, 8),
                                np.linspace(5e-2, 1-5e-2, 16)[1:-1],
                                1. - np.linspace(5e-2, -1e-6, 8),])

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
    overconfidence_thresh = 0.3 / 100.
    confident_pred = (max_pred > (1 - overconfidence_thresh))

    # 'expected calibration error', see weinberger paper on calibration
    ece = np.sum(np.abs(q-p) * hist_tot) / np.sum(hist_tot)
    mce = np.max(np.abs(q-p))
    ice = np.trapz(np.abs(q-p), x=p)
    ovc = np.mean((1 - correct_pred)[confident_pred]) /  overconfidence_thresh
    #ovc = (1. - np.mean(q[-2:])) / (1. - pred_bins[-2])
    #print('>> confidence thresshold', pred_bins[-2])
    return ece, mce, ice, ovc

