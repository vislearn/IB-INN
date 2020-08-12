import torch
import numpy as np

def test_metrics(inn, data, args):
    metrics = {'accuracy':     [],
               'bits_per_dim': [],
               'L_x':          [],
               'L_y':          []}


    with torch.no_grad():
        for x, y in data.test_loader:
            x, y = x.cuda(), data.onehot(y.cuda())
            output = inn.validate(x, y)

            metrics['L_x'].append(output['nll_joint_val'].item())
            metrics['L_y'].append(output['cat_ce_val'].item())

            bpd = output['nll_joint_val'].item()
            if eval(args['data']['dequantize_uniform']):
                bpd += np.log(256)
            else:
                bpd += np.log(data.sigma) + 0.5 * np.log(2 * np.pi)
            bpd -= 0.5 * np.log(2*np.pi)
            bpd /= np.log(2)

            metrics['bits_per_dim'].append(bpd)
            metrics['accuracy'].append(output['acc_val'].item())

    for k in metrics:
        # has to be cast from np.float32 to float() explicitly,
        # otherwise json will not accept it.
        metrics[k] = float(np.mean(metrics[k]))

    return metrics

