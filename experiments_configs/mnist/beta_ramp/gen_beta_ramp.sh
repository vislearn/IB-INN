beta_values=(   0.0158 
                0.0201 
                0.0255 
                0.0323 
                0.0409 
                0.0518 
                0.0656 
                0.0832 
                0.1054 
                0.1336 
                0.1693 
                0.2145 
                0.2718 
                0.3445 
                0.4365 
                0.5532 
                0.7010 
                0.8883 
                1.1257 
                1.4265 
                1.8078 
                2.2909 
                2.9031 
                3.6789 
                4.6620 
                5.9078 
                7.4866 
                9.4873 
                12.0226 
                15.2355 
                19.3070 
                24.4665 
                31.0048 
                39.2903 
                49.7901 
                63.0957
                )


for b in ${beta_values[@]}; do
    cat << EOF > beta_${b}.ini
[checkpoints]
base_name = beta_${b}_mnist
interval_checkpoint = 200
interval_figure = 200
interval_log = 240

[training]
beta_IB = ${b}
n_epochs = 60
scheduler_milestones = [50]

[data]
dataset = MNIST
label_smoothing = 0.01

[model]
act_norm = 0.80
EOF
done
