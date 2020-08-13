import glob
import json
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pattern = 'output/sigma_*/results.json'
result_jsons = []

for f in glob.glob(pattern):
    results = json.load(open(f))
    m = re.search('[0-9.]+', f)
    results['sigma'] = float(m.group(0))
    result_jsons.append(results)
result_jsons.sort(key = (lambda d: d['sigma']))
print(result_jsons[0])
#assert False

sig = np.array([c['sigma'] for c in result_jsons])
lx  = np.array([c['test_metrics']['L_x'] for c in result_jsons])
ly  = np.array([c['test_metrics']['L_y'] for c in result_jsons])
acc = np.array([c['test_metrics']['accuracy'] for c in result_jsons])
ce  = np.array([c['calib_err']['mce'] for c in result_jsons])
ood = sum(np.array([c['ood_tt'][k] for c in result_jsons]) for k in ['quickdraw', 'rot_rgb', 'noisy'])
ood /= 3.

plots_h = 1
plots_w = 3

plt.figure(figsize=(plots_w * 4.2, plots_h * 3.5))

def subplot_init(i):
    ax = plt.subplot(plots_h, plots_w, i + 1)
    plt.xlim(1e-4, 1)
    plt.grid(True, alpha=0.45)
    #plt.xlabel('$\\sigma$')

def verty(x, **kwargs):
    ymin, ymax = plt.ylim()
    plt.semilogx([x, x], [ymin, ymax], **kwargs)
    plt.ylim(ymin, ymax)

quant_label = 'Quantization $\\Delta X$'
subplot_init(0)
plt.semilogx(sig, lx, label='$\\mathcal{L}_X$', color='darkviolet')
plt.semilogx(sig, ly, label='$-\\mathcal{L}_Y$', color='darkturquoise')
verty(1/256, color='gray', linestyle='dashed', label=quant_label)
plt.ylabel('Test loss')
plt.legend()

subplot_init(1)
plt.semilogx(sig, 100*acc, label='Accuracy', color='royalblue')
plt.semilogx(sig, 100*ood, label='OoD detection score', color='forestgreen')
plt.semilogx(sig, ce, label='Calib. err.', color='darkorange')
plt.ylabel('Performance measure (in %)')
verty(1/256, color='gray', linestyle='dashed', label=quant_label)
plt.legend()


subplot_init(2)
ci = lx - np.log(sig) - 0.5 * np.log(2. * np.pi * np.e)
sig_mi = np.logspace(-4, -3)
mi = -4.3 - np.log(sig_mi)

ci *= 3072
mi *= 3072

plt.semilogx(sig, ci, label='Approx. $CI(X, Z_\\varepsilon)$', color='darkviolet')
plt.semilogx(sig_mi, mi, label='$I(X, Z_\\varepsilon) + const., \\sigma \\to 0$', color='black')
verty(1/256, color='gray', linestyle='dashed', label=quant_label)
plt.ylabel('Information (in nats)')
plt.legend()

plt.tight_layout()
plt.savefig('figures/sigma_ablation.pdf')
