#!/bin/bash

#SBATCH --job-name              struct2
#SBATCH --time                  48:00:00
#SBATCH --cpus-per-task         6
#SBATCH --gres                  gpu:1
#SBATCH --mem                   200G
#SBATCH --output=output/struct2.out
#SBATCH --partition             a100_batch

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


# K=16
# L=5
# R=1
# CUDA_VISIBLE_DEVICES=0 \
# python main.py --cfg configs/GSE/ogbg/molpcba-GSE_grit-Poly.yaml  \
# posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
# gse_model.messaging.num_blocks $((L)) \
# gse_model.messaging.repeats $((R)) \
# gse_model.full.repeats $((R)) \
# name_tag mixed_bern_K$((K))L$((L))R$((R))


# K=16
# L=5
# R=2
# CUDA_VISIBLE_DEVICES=0 \
# python main.py --cfg configs/GSE/ogbg/molpcba-GSE_grit-Poly.yaml  \
# posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
# gse_model.messaging.num_blocks $((L)) \
# gse_model.messaging.repeats $((R)) \
# gse_model.full.repeats $((R)) \
# name_tag mixed_bern_K$((K))L$((L))R$((R))


# ====== Task: 25652 ======
# K=16
# L=5
# R=1
# CUDA_VISIBLE_DEVICES=0 \
# python main.py --cfg configs/GSE/ogbg/molpcba-GSE_grit-Poly.yaml  \
# posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
# gse_model.messaging.num_blocks $((L)) \
# gse_model.messaging.repeats $((R)) \
# gse_model.full.repeats $((R)) \
# gse_model.hidden_dim 384 \
# gse_model.attn_heads 24 \
# name_tag mixed_bern_K$((K))L$((L))R$((R))_0523


# # ====== Task: 25657 ======
# CUDA_VISIBLE_DEVICES=0 \
# python main.py --repeat 5 --cfg configs/GSE/bench_cifar10/cifar10-GSE_grit-Poly.yaml name_tag M0F3 & 

# CUDA_VISIBLE_DEVICES=0 \
# python main.py --repeat 5 --cfg configs/GSE/bench_cluster/cluster-GSE_grit-Poly.yaml M6r2Fr4 & 

# wait 

# # ====== Task: 25658 ======
# CUDA_VISIBLE_DEVICES=0 \
# python main.py --cfg configs/GSE/bench_mnist/mnist-GSE_grit-Poly.yaml name_tag M0F3 & 

# CUDA_VISIBLE_DEVICES=0 \
# python main.py --cfg configs/GSE/bench_pattern/pattern-GSE_grit-Poly.yaml M6r1Fr3 & 

# wait 


# ========= Task 25877 =====================
# K=8
# python main.py --cfg configs/GSE/peptides/peptides_struct-GSE_grit-full.yaml seed 4 \
# posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
# name_tag LowMiddleK$((K))

# K=16
# python main.py --cfg configs/GSE/peptides/peptides_struct-GSE_grit-full.yaml seed 4 \
# posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
# name_tag LowMiddleK$((K))


# K=12
# python main.py --cfg configs/GSE/peptides/peptides_struct-GSE_grit-full.yaml seed 4 \
# posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
# name_tag LowMiddleK$((K))


# ========= Task 25876 =====================

# K=8
# python main.py --cfg configs/GSE/peptides/peptides_func-GSE_grit-Poly-full.yaml seed 4 \
# posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
# name_tag LowMiddleK$((K))


# K=12
# python main.py --cfg configs/GSE/peptides/peptides_func-GSE_grit-Poly-full.yaml seed 4 \
# posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
# name_tag LowMiddleK$((K))


# K=16
# python main.py --cfg configs/GSE/peptides/peptides_func-GSE_grit-Poly-full.yaml seed 4 \
# posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
# name_tag LowMiddleK$((K))



# # ============ Task - 25879 ============
# K=8
# python main.py --cfg configs/GSE/peptides/peptides_func-GSE_grit-Poly-full.yaml seed 4 \
# posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
# gse_model.attn_drop_prob 0.2 \
# name_tag LowMiddleK$((K))


# K=12
# python main.py --cfg configs/GSE/peptides/peptides_func-GSE_grit-Poly-full.yaml seed 4 \
# posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
# gse_model.attn_drop_prob 0.2 \
# name_tag LowMiddleK$((K))


# K=16
# python main.py --cfg configs/GSE/peptides/peptides_func-GSE_grit-Poly-full.yaml seed 4 \
# posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
# gse_model.attn_drop_prob 0.2 \
# name_tag LowMiddleK$((K))



# ========= Task 25880 =====================
K=8
python main.py --cfg configs/GSE/peptides/peptides_struct-GSE_grit-full.yaml seed 4 \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.attn_drop_prob 0.1 \
gse_model.drop_prob 0.0 \
name_tag LowMiddleK$((K))

K=16
python main.py --cfg configs/GSE/peptides/peptides_struct-GSE_grit-full.yaml seed 4 \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.attn_drop_prob 0.1 \
gse_model.drop_prob 0.0 \
name_tag LowMiddleK$((K))


K=12
python main.py --cfg configs/GSE/peptides/peptides_struct-GSE_grit-full.yaml seed 4 \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.attn_drop_prob 0.1 \
gse_model.drop_prob 0.0 \
name_tag LowMiddleK$((K))


echo "finished!!!"
