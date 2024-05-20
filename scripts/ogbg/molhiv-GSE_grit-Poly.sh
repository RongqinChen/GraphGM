
K=4
L=3
R=1
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/GSE/ogbg/molhiv-GSE_grit-Poly.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.messaging.repeats $((R)) \
gse_model.full.repeats $((R)) \
name_tag mixed_bern_K$((K))L$((L))R$((R))

wait


K=8
L=4
R=1
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/GSE/ogbg/molhiv-GSE_grit-Poly.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.messaging.repeats $((R)) \
gse_model.full.repeats $((R)) \
name_tag mixed_bern_K$((K))L$((L))R$((R))

wait

