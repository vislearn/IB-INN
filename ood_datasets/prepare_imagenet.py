from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import ImageCollection
from skimage.transform import resize
from tqdm import tqdm

orig_img_pattern = '/home/diz/data/imagenet/val/*.JPEG'
dataset_size = 8_000

images = ImageCollection(orig_img_pattern)

imgs_downsampled = []
for i in tqdm(range(dataset_size)):
    im = images[i]

    if len(im.shape) == 2:
        im = np.stack([im]*3, axis=2)
    if im.shape[2] == 4:
        im = im[:3]

    h, w, _ = im.shape
    dw = min(im.shape[0], im.shape[1])
    i = int(round((h - dw) / 2.))
    j = int(round((w - dw) / 2.))
    im = im[i:i+dw, j:j+dw]

    im = resize(im, (32,32), order=3)
    imgs_downsampled.append(im.transpose((2,0,1)))

imgs_downsampled = np.array(imgs_downsampled)
np.save('./ood_datasets/imagenet.npy', imgs_downsampled)

print(imgs_downsampled.shape)
for k in range(100):
    plt.subplot(10,10,k+1)
    plt.imshow(imgs_downsampled[k].transpose((1,2,0)))

plt.tight_layout()
plt.savefig('examples.pdf')

