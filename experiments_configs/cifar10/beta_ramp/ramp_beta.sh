config_orig=default.ini
config_new=tmp.ini
data_file=overconfidence_beta.dat

variables=(
    base_name
    beta_IB
)

replace () {
    cp $config_orig $config_new

    for v in "${variables[@]}"; do
        sed -ie "s/${v} *=\(.*\)/${v} = $1/" $config_new
        shift
    done
}

betas=(
    0.005
    0.006
    0.007
    0.008
    0.010
    0.012
    0.014
    0.017
    0.020
    0.024
    0.028
    0.033
    0.040
    0.047
    0.056
    0.067
    0.079
    0.094
    0.112
    0.133
    0.158
    0.188
    0.224
    0.266
    0.316
    0.376
    0.447
    0.531
    0.631
    0.750
    0.891
    1.059
    1.259
    1.496
    1.778
    2.113
    2.512
    2.985
    3.548
    4.217
    5.012
    5.957
    7.079
    8.414
    10.000
    15.000
    20.000
)

for b in "${betas[@]}"; do
    replace beta_${b} $b
    #python main.py train tmp.ini
    python main.py test tmp.ini
done

rm tmp.ini

> $data_file

for b in "${betas[@]}"; do
    sed -ne "/overconfidence/{s/overconfidence/${b} /p}" output/beta_${b}/results.txt >> /tmp/$data_file
done

# add the test accuracy to the data file also
for b in "${betas[@]}"; do
    sed -ne "/acc/{s/acc//p}" output/beta_${b}/results.txt
done | paste /tmp/$data_file - > $data_file

# plot the data using matplotlib
python << END
import numpy as np
import matplotlib.pyplot as plt

dat = np.loadtxt("${data_file}").T
plt.subplot(2, 1, 1)
plt.plot(np.log(dat[0]), 1.-dat[-1], '-o')
plt.xticks(np.log(dat[0, ::3]), ['%.3f'%b if b<0.01 else '%.2f'%b for b in dat[0, ::3] ])
plt.ylabel('error rate')
plt.xlabel('beta')
plt.grid(True)

plt.subplot(2, 1, 2)
for k in range(3, 4):
    plt.plot(np.log(dat[0]), dat[k], '-')
plt.xticks(np.log(dat[0, ::3]), ['%.3f'%b if b<0.01 else '%.2f'%b for b in dat[0, ::3] ])
plt.ylabel('overconfidence (thresshold 1E-3)')
plt.xlabel('beta')
plt.grid(True)

plt.tight_layout()
plt.show()
END
