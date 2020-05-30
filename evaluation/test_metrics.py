import torch
import numpy as np

def test_metrics(inn, data):
    acc = []
    bits_per_dim = []

    with torch.no_grad():
        for x, y in data.test_loader:
            x, y = x.cuda(), data.onehot(y.cuda())
            output = inn.validate(x, y)


            bpd = output['nll_joint_val'].item()
            bpd += np.log(256)
            bpd /= np.log(2)

            bits_per_dim.append(bpd)
            acc.append(output['acc_val'].item())

    return np.mean(acc), np.mean(bits_per_dim)

