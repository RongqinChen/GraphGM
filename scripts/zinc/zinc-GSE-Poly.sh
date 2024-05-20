
K=4
L=3
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/zinc/zinc-GSE.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
name_tag mixed_bern_K$((K))L$((L)) &

wait


K=6
L=3
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/zinc/zinc-GSE.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
name_tag mixed_bern_K$((K))L$((L)) &

wait


K=8
L=4
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/zinc/zinc-GSE.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
name_tag mixed_bern_K$((K))L$((L)) &

wait


K=10
L=4
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/zinc/zinc-GSE.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
name_tag mixed_bern_K$((K))L$((L)) &

wait
