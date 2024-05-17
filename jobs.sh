#!/bin/bash

#SBATCH --job-name              clu-grit-mixed_bern
#SBATCH --time                  48:00:00
#SBATCH --cpus-per-task         6
#SBATCH --gres                  gpu:1
#SBATCH --mem                   200G
#SBATCH --output=results/clu-grit-mixed_bern.out
#SBATCH --partition             h800_batch

your_cleanup_function()
{
    echo "function your_cleanup_function called at $(date)" # do whatever cleanup you want here
    pkill -u rongqin
}

# call your_cleanup_function once we receive USR1 signal 
trap 'your_cleanup_function' USR1


source /home/yc07917/mambaforge/bin/activate gnn210
which python

free -h
echo ""
lscpu
echo ""
nvidia-smi
echo ""


K=30

CUDA_VISIBLE_DEVICES=0 \
    python main.py --repeat 5 --cfg configs/GSE/cluster/cluster-GT-GRIT-Poly.yaml  \
    posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
    name_tag mixed_bern_K$((K)) &

wait


echo "finished!!!"
