extract_numb () {
    echo "$*" | grep -oEe '-?[0-9]+\.[0-9]+'
}

write_all_to_txt () {

    for dir in $(ls -d $1); do
        beta=$(extract_numb ${dir})
        results=$(cat ${dir}/results.dat | sed -e '/LATEX/d;/DATASET/d;s/[A-Z]//g' | tr -d '\n' | sed -e 's/  */ /g')
        echo $beta $results
    done | sort -n | tee $2
}

write_all_to_txt './scratch_mnt/beta_*.????' ./beta_curves/ibinn_cifar10.txt
write_all_to_txt './scratch_mnt/vib_beta_*.????' ./beta_curves/vib_cifar10.txt
