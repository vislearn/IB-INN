import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from model import GenerativeClassifier


def show_samples(model, data, y, T=0.75):
    with torch.no_grad():
        samples = model.sample(y, T)
        samples = data.de_augment(samples).cpu().numpy()
        samples = np.clip(samples, 0, 1)

    w = min(y.shape[1], 10)
    h = int(np.ceil(y.shape[0] / w))

    plt.figure()
    for k in range(y.shape[0]):
        plt.subplot(h, w, k + 1)
        if data.dataset == 'MNIST':
            plt.imshow(samples[k], cmap='gray')
        else:
            plt.imshow(samples[k].transpose(1, 2, 0))
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
    true_label = np.concatenate(true_label, axis=0)

    plt.figure()
    plt.scatter(mu_red[:, 0], mu_red[:, 1], c=np.arange(data.n_classes), cmap='tab10', s=250, alpha=0.5)
    plt.scatter(z_red[:, 0], z_red[:, 1], c=true_label, cmap='tab10', s=1)
    plt.tight_layout()
