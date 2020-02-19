import configparser

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ood_datasets.cifar
import ood_datasets.quickdraw
import ood_datasets.imagenet

args = configparser.ConfigParser()
args.read('default.ini')
n = 3
dn = 8

for loader, name, dn in [ (ood_datasets.cifar.cifar_rgb_rotation(args, 0.35), 'rot', 35),
                          (ood_datasets.cifar.cifar_noise(args, 0.01), 'noise', 35),
                          (ood_datasets.quickdraw.quickdraw_colored(args), 'draw', 0),
                          (ood_datasets.imagenet.imagenet(args), 'imagenet', 7),]:


        plt.figure(figsize=(1.5 * n, 1.5))
        for x in loader:
            x = ood_datasets.cifar.default_augment.de_augment(x[0])
            for j in range(n):
                plt.subplot(1, n, 1 + j)
                plt.imshow(x[j+dn].numpy().transpose((1,2,0)))
                plt.xticks([])
                plt.yticks([])
            break
        plt.tight_layout(pad=0.0, h_pad=0.1, w_pad=0.1)
        plt.savefig(f'./figures/ood_data_example_{name}.pdf')
