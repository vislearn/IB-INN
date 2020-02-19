for i in {0..9}; do
    cat << EOF > full_model_ensemb_${i}.ini
[checkpoints]
base_name = full_model_beta1
ensemble_index = $i
interval_checkpoint = 1000
interval_figure = 200
live_updates = False

[data]
noise_amplitde = 0.005

[training]
beta_IB = 1.0
n_epochs = 450
clip_grad_norm = 8.
scheduler_milestones = [150, 250, 350]
optimizer = SGD
lr = 0.07
EOF
done
