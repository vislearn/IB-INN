import torch
import numpy as np

def test_metrics(inn, data):
    acc = []
    with torch.no_grad():
        for x, y in data.test_loader:
            x, y = x.cuda(), data.onehot(y.cuda())
            acc.append(inn.validate(x, y)['acc_val'].item())

    return np.mean(acc)

