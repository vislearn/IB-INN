cat << EOF > /tmp/ood 
0.000   0.50000
0.043   0.53157
0.087   0.58003
0.130   0.62740
0.174   0.67038
0.217   0.71105
0.261   0.75318
0.304   0.79818
0.348   0.84194
0.391   0.87938
0.435   0.90850
0.478   0.93016
0.522   0.94711
0.565   0.96007
0.609   0.96912
0.652   0.97552
0.696   0.98011
0.739   0.98339
0.783   0.98568
0.826   0.98732
0.870   0.98830
0.913   0.98855
0.957   0.98817
1.000   0.98704
EOF

cat << 'EOF' | python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.loadtxt('/tmp/ood').T
alpha, auc = data

plt.figure(figsize=(8,5.0))
plt.plot(alpha, auc, '-o')
plt.ylim(0.5, 1.0)
plt.xlim(0., 1.)
plt.ylabel('AUC')
plt.grid(True, alpha=0.45)
plt.tight_layout()
plt.savefig('figures/auc_ood_interpolation.png', dpi=240)
EOF

python -m ood_datasets.cifar
montage figures/auc_ood_interpolation.png figures/data_ood_interpolation.png -tile 1x2 -geometry +0+0 figures/ood_interpolation.png
