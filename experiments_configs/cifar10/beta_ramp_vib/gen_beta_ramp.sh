beta_values=(0.0200
             0.0280
             0.0394
             0.0554
             0.0779
             0.1094
             0.1538
             0.2162
             0.3038
             0.4270
             0.6002
             0.8435
             1.1855
             1.6662
             2.3419
             3.2915
             4.6261
             6.5019
             9.1384
             12.8439
             18.0519
             25.3716
             35.6594
             50.1187)


for b in ${beta_values[@]}; do
    cat << EOF > beta_${b}.ini
[checkpoints]
base_name = vib_beta_${b}
interval_checkpoint = 500
interval_figure = 200
live_updates = False

[training]
beta_IB = ${b}

[ablations]
vib = True
EOF
done
