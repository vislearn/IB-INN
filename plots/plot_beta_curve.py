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
colors = {'ibinn': ['royalblue', 'forestgreen', 'darkorange', 'royalblue','royalblue','royalblue'],
          'vib': ['royalblue', 'forestgreen', 'darkorange', 'royalblue','royalblue','royalblue']}
alphas = {'ibinn':1., 'vib':0.65}
model_labels = {'ibinn': 'IB-INN', 'vib': 'VIB'}

beta_ticks = np.linspace(-1.5, 1.5, 5)
beta_labels = [('%.5f' % (10**b))[:4] for b in beta_ticks]
#beta_ticks = np.sort(np.array(list(set(data_ibinn[0]).union(set(data_vib[0])))+[50,50,50]))[::4]
#beta_ticks = np.log10(beta_ticks)

plt.figure(figsize=(14, 3.3))
plots_h = 1
plots_w = 4

titles = ['Classification error ($\\downarrow$)',
          'Geom. mean of calibration errors ($\\downarrow$)',
          'OoD pred. entropy ($\\uparrow$)',
          'OoD sep. RGB rotation ($\\uparrow$)',
          'OoD sep. Drawings ($\\uparrow$)',
          'OoD sep. Noisy CIFAR ($\\uparrow$)']

for i, plot in enumerate(['acc_err', 'xce', 'ent4']):
    ax = plt.subplot(plots_h,plots_w,i+1)
    ax.set_title(titles[i])
    plt.xticks(beta_ticks, beta_labels)
    #plt.xlabel('$\\tilde \\beta$')
    if i == 1:
        plt.yticks(np.arange(7), ['%.1f' % e for e in np.arange(7)])

    for model in models:
        data = curves[model]
        beta = data[cols['beta']]
        logbeta = np.log10(beta)

        if i == 2 and model == 'ibinn':
            label = 'ImageNet'
        else:
            label = model_labels[model]

        plt.plot(logbeta, data[cols[plot]],
                 line_types[model], color=colors[model][i], alpha=alphas[model],
                 label=label)

        if i == 2 and model == 'ibinn':
            plt.legend()

    if i == 0:
        plt.legend()


    plt.grid(True, alpha=0.45)
    plt.xlim(np.log10(0.02), np.log10(50))

ax = plt.subplot(plots_h, plots_w, 4)
ax.set_title('OoD detection score ($\\uparrow$)')
auc_labels = {'auc1': 'RGB rot.',
              'auc2':'QuickDraw',
              'auc3':'Noisy',
              'auc4': 'ImageNet'}

auc_colors = {'auc1': 'lightcoral',
              'auc2':'maroon',
              'auc3':'indianred'}

auc_linetypes = {'auc1': '-',
                 'auc2':'-.',
                 'auc3':'--'}

beta = data_ibinn[cols['beta']]
logbeta = np.log10(beta)

for auc in ['auc1', 'auc3', 'auc2']:
    plt.plot(logbeta, data_ibinn[cols[auc]],
              auc_linetypes[auc],
             color=auc_colors[auc],
             label=auc_labels[auc])
plt.legend()
plt.grid(True, alpha=0.45)
plt.xlim(np.log10(0.02), np.log10(50))
plt.xticks(beta_ticks, beta_labels)
#plt.xlabel('$\\tilde \\beta$')

plt.tight_layout(w_pad = -2)
plt.savefig('./figures/beta.pdf')
