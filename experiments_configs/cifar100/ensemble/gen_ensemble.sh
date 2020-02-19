for i in {0..7}; do
    cat << EOF > full_model_ensemb_${i}.ini
[checkpoints]
base_name = full_model_cifar100
ensemble_index = $i
interval_figure = 1000

[training]
beta_IB = 1.0

[data]
dataset = CIFAR100

EOF
done
