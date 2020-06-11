import glob
import json
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['ibinn', 'vib']

patterns = {'ibinn': 'output/beta_*/results.json',
            'vib':   'output/vib_beta_*/results.json'}

curves = {'ibinn':[],
          'vib': []}

for mod in models:
    for f in glob.glob(patterns[mod]):
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
print(beta_axis['vib'])

line_types = {'ibinn': '-',
              'vib': ':'}

alphas = {'ibinn':1., 'vib':0.65}
model_labels = {'ibinn': 'IB-INN', 'vib': 'VIB'}

beta_ticks = np.linspace(-1.5, 1.5, 5)
beta_labels = [('%.5f' % (10**b))[:4] for b in beta_ticks]

plt.figure(figsize=(18, 11))
plots_h = 3
plots_w = 5

titles = ['Classification error ($\\downarrow$)',
          'Geom. mean of calibration errors ($\\downarrow$)',
          'Expected calibration error ($\\downarrow$)',
          'Maximum calibration error ($\\downarrow$)',
          'Integrated calibration error ($\\downarrow$)',

          'Average incr. OoD pred. ent. ($\\uparrow$)',
          'incr. OoD pred. ent. - RGB rot. ($\\uparrow$)',
          'incr. OoD pred. ent. - Noisy ($\\uparrow$)',
          'incr. OoD pred. ent. - QuickDraw ($\\uparrow$)',
          'incr. OoD pred. ent. - ImageNet ($\\uparrow$)',

          'Average OoD detection score ($\\uparrow$)',
          'OoD detection score - RGB rot. ($\\uparrow$)',
          'OoD detection score - Noisy ($\\uparrow$)',
          'OoD detection score - QuickDraw ($\\uparrow$)',
          'OoD detection score - ImageNet ($\\uparrow$)',
          ]

colors = ['royalblue'] + ['darkorange']*4 + ['maroon']*5 + ['forestgreen']*5

plots = [('test_metrics', 'acc'), ('calib_err', 'gme'),   ('calib_err', 'ece'),   ('calib_err', 'mce'),   ('calib_err', 'ice'),
         ('ood_d_ent', 'ari_mean'), ('ood_d_ent', 'rot_rgb'), ('ood_d_ent', 'noisy'), ('ood_d_ent', 'quickdraw'), ('ood_d_ent', 'imagenet'),
         ('ood_tt', 'ari_mean'),  ('ood_tt', 'rot_rgb'),  ('ood_tt', 'noisy'),  ('ood_tt', 'quickdraw'),  ('ood_tt', 'imagenet')
        ]

for i, plot in enumerate(plots):
    ax = plt.subplot(plots_h,plots_w,i+1)
    ax.set_title(titles[i])
    plt.xticks(beta_ticks, beta_labels)

    for model in models:
        data = np.array([c[plot[0]][plot[1]] for c in curves[model]])
        if plot[1] == 'acc':
            data = 100. - data
        if plot[0] == 'ood_tt':
            data *= 100.

        if model == 'vib' and plot[0] == 'ood_tt':
            continue

        plt.plot(beta_log_axis[model], data,
                 line_types[model], color=colors[i], alpha=alphas[model],
                 label=model_labels[model])
        if plot[0] == 'ood_tt':
            plt.ylim(45, 100)
            plt.plot(beta_log_axis[model], 50 + 0.*data, '-', color='black', label='Random')
        if plot[1] == 'ece':
            plt.ylim(0.3, 1.5)
        if plot[0] == 'ood_ent':
            plt.ylim(0.0, 1.25)

    #if i == 0:
        #plt.legend()

    plt.grid(True, alpha=0.45)
    plt.xlim(np.log10(0.02), np.log10(55))

plt.tight_layout(w_pad = -2)
plt.savefig('./figures/beta_appendix.pdf')
