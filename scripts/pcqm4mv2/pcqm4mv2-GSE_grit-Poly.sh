
# K=4
# L=3
# CUDA_VISIBLE_DEVICES=0 \
# python main.py --cfg configs/GSE/pcqm4mv2/pcqm4mv2-GSE_grit-Poly.yaml  seed 4 \
# posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
# gse_model.messaging.num_blocks $((L)) \
# gse_model.hidden_dim 72 \
# name_tag mixed_bern_K$((K))L$((L)) &

# # wait


# K=8
# L=4
# CUDA_VISIBLE_DEVICES=1 \
# python main.py --cfg configs/GSE/pcqm4mv2/pcqm4mv2-GSE_grit-Poly.yaml  seed 4 \
# posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
# gse_model.messaging.num_blocks $((L)) \
# name_tag mixed_bern_K$((K))L$((L)) &

# # wait


K=16
L=5
R=1
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/GSE/pcqm4mv2/pcqm4mv2-GSE_grit-Poly.yaml  seed 4 \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.messaging.repeats $((R)) \
gse_model.full.repeats $((R)) \
gse_model.hidden_dim 72 \
name_tag mixed_bern_K$((K))L$((L))R$((R)) &

# # wait

# K=32
# L=6
# R=1
# CUDA_VISIBLE_DEVICES=0 \
# python main.py --cfg configs/GSE/pcqm4mv2/pcqm4mv2-GSE_grit-Poly.yaml  seed 4 \
# posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
# gse_model.messaging.num_blocks $((L)) \
# gse_model.messaging.repeats $((R)) \
# gse_model.hidden_dim 64 \
# name_tag mixed_bern_K$((K))L$((L))R$((R)) &

# wait
