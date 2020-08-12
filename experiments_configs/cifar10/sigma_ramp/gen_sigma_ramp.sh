sigma_values=(0.000100
              0.000147
              0.000215
              0.000316
              0.000464
              0.000681
              0.001000
              0.001468
              0.002154
              0.003162
              0.004642
              0.006813
              0.010000
              0.014678
              0.021544
              0.031623
              0.046416
              0.068129
              0.100000
              0.146780
              0.215443
              0.316228
              0.464159
              0.681292
              1.000000)


for s in ${sigma_values[@]}; do
    cat << EOF > sigma_${s}.ini
[checkpoints]
base_name = sigma_${s}
interval_checkpoint = 1000
interval_figure = 1000
live_updates = False

[training]
beta_IB = 0.2

[data]
noise_amplitde = ${s}
dequantize_uniform = False
EOF
done
