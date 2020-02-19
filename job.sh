gpu=$2
cores=$3

cat << EOF > jobs/$(basename -- $1).sh
#!/bin/sh
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --partition=$gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cores
#SBATCH --mem-per-cpu=2048
#SBATCH -J "$(basename -- $1)"
#SBATCH -A p_hpdlf_itwm
#SBATCH -e jobs/out/$(basename -- $1).err
#SBATCH -o jobs/out/$(basename -- $1).out

module load modenv/scs5
module load CUDA
module load cuDNN
module load Anaconda3

source activate test

python -W ignore main.py train $1
#python -W ignore main.py test $1
EOF
