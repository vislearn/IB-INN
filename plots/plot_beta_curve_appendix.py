import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_ibinn = np.loadtxt('./beta_curves/ibinn_cifar10.txt').T
data_vib = np.loadtxt('./beta_curves/vib_cifar10.txt').T

cols = {}
for i, c in enumerate(['beta', 'acc', 'acc_err', 'ece', 'mce', 'oce', 'xce', 'xce_o', 'auc1', 'auc2',
                       'auc3', 'auc4', 'auc_err1', 'auc_err2', 'auc_err3', 'auc_err4', 'ent1', 'ent2',
                       'ent3', 'ent4', 'auc_gm', 'ent_gm']):
    cols[c] = i

models = ['ibinn', 'vib']

curves = {'ibinn': data_ibinn,
          'vib':data_vib}
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
          'Overconfidence ($\\downarrow$)',

          'Geom. mean OoD pred. entropy ($\\uparrow$)',
          'OoD pred. entropy - RGB rot. ($\\uparrow$)',
          'OoD pred. entropy - Noisy ($\\uparrow$)',
          'OoD pred. entropy - QuickDraw ($\\uparrow$)',
          'OoD pred. entropy - ImageNet ($\\uparrow$)',

          'Geom. mean OoD detection score ($\\uparrow$)',
          'OoD detection score - RGB rot. ($\\uparrow$)',
          'OoD detection score - Noisy ($\\uparrow$)',
          'OoD detection score - QuickDraw ($\\uparrow$)',
          'OoD detection score - ImageNet ($\\uparrow$)',
          ]

colors = ['royalblue'] + ['forestgreen']*4 + ['darkorange']*5 + ['maroon']*5

plots = ['acc_err', 'xce', 'ece', 'mce', 'oce',
         'ent_gm', 'ent1','ent2','ent3','ent4',
         'auc_gm', 'auc1','auc2','auc3','auc4']

for i, plot in enumerate(plots):
    ax = plt.subplot(plots_h,plots_w,i+1)
    ax.set_title(titles[i])
    plt.xticks(beta_ticks, beta_labels)

    for model in models:
        data = curves[model]
        beta = data[cols['beta']]
        logbeta = np.log10(beta)

        plt.plot(logbeta, data[cols[plot]],
                 line_types[model], color=colors[i], alpha=alphas[model],
                 label=model_labels[model])

    if i == 0:
        plt.legend()

    plt.grid(True, alpha=0.45)
    plt.xlim(np.log10(0.02), np.log10(50))

plt.tight_layout(w_pad = -2)
plt.savefig('./figures/beta_appendix.pdf')
