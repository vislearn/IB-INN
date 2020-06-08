import glob
import json
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['ibinn', 'vib']

patterns = {'ibinn': ['output/beta_*/results.json'],
            'vib':   ['output/vib_beta_?.????/results.json',
                      'output/vib_beta_??.????/results.json']}

curves = {'ibinn':[],
          'vib': []}

for mod in models:
    for pat in  patterns[mod]:
        for f in glob.glob(pat):
            results = json.load(open(f))
            m = re.search('[0-9.]+', f)
            results['beta'] = float(m.group(0))

            curves[mod].append(results)
    curves[mod].sort(key = (lambda d: d['beta']))
    print(mod, len(curves[mod]))

# these models did not converge:
# (constant 10% acc)
curves['vib'] = curves['vib'][3:]

beta_axis = {m : [c['beta'] for c in curves[m]] for m in models}
beta_log_axis = {m: np.log10(beta_axis[m]) for m in models}

line_types = {'ibinn': '-',
              'vib': ':'}

alphas = {'ibinn':1., 'vib':0.65}
model_labels = {'ibinn': 'IB-INN', 'vib': 'VIB'}

beta_ticks = np.linspace(-1.5, 1.5, 5)
beta_labels = [('%.5f' % (10**b))[:4] for b in beta_ticks]

plt.figure(figsize=(14, 3.3))
plots_h = 1
plots_w = 4

titles = ['Classification error ($\\downarrow$)',
          'Geom. mean of calibration errors ($\\downarrow$)',
          'Incr. OoD pred. entropy ($\\uparrow$)',
          'OoD detection score ($\\uparrow$)',
          ]

def get_curve(model, category, entry):
    return (beta_log_axis[model],
            np.array([c[category][entry] for c in curves[model]]))

def subplot_init(i):
    ax = plt.subplot(plots_h,plots_w,i+1)
    ax.set_title(titles[i])
    plt.xlim(np.log10(0.02), np.log10(55))
    plt.xticks(beta_ticks, beta_labels)
    plt.grid(True, alpha=0.45)

subplot_init(0)
for m in models:
    b, a = get_curve(m, 'test_metrics', 'acc')
    a = 100. - a
    plt.plot(b, a, line_types[m], alpha=alphas[m], color='royalblue', label=model_labels[m])
plt.legend()

subplot_init(1)
for m in models:
    b, a = get_curve(m, 'calib_err', 'gme')
    plt.plot(b, a, line_types[m], alpha=alphas[m], color='darkorange', label=model_labels[m])
plt.legend()

subplot_init(2)
                #["#173a1e","#1e5226","#1e661e","#3d7f2f"],
for data, label, color, linetype in zip(['rot_rgb', 'noisy', 'quickdraw', 'imagenet'],
                                        ['RGB-rot.', 'Noise', 'QuickDraw', 'ImageNet'],
                                        ["#4c0404","#911818","#cb4343","#e28d8d"],
                                        ['solid', (0,(3,1)), (0,(3,1,1,1)), (0, (7,2))]
                               ):
    b, a = get_curve('ibinn', 'ood_d_ent', data)
    plt.plot(b, a, label=label, color=color, linestyle=linetype)

plt.legend()
plt.ylim(0, 0.7)

subplot_init(3)
for data, label, color, linetype in zip(['rot_rgb', 'noisy', 'quickdraw', 'imagenet'],
                                        ['RGB-rot.', 'Noise', 'QuickDraw', 'ImageNet'],
                                        ["#0a3d0a","#1b791b","#52b040","#84de6e"],
                                        ['solid', (0,(3,1)), (0,(3,1,1,1)), (0, (7,2))]
                               ):

    b, a = get_curve('ibinn', 'ood_tt', data)
    plt.plot(b, 100. * a, label=label, color=color, linestyle=linetype)
plt.ylim(50, 100)
plt.legend()

plt.tight_layout(w_pad = -2)
plt.savefig('./figures/beta_new.pdf')
