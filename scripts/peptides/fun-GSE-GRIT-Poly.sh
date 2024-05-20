
K=4
L=3
F=1
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/peptides/peptides_func-GSE-Poly.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.full.repeats $((F)) \
name_tag mixed_bern_K$((K))L$((L))F$((F)) &

wait


K=6
L=3
F=1
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/peptides/peptides_func-GSE-Poly.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.full.repeats $((F)) \
name_tag mixed_bern_K$((K))L$((L))F$((F)) &

wait


K=8
L=4
F=1
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/peptides/peptides_func-GSE-Poly.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.full.repeats $((F)) \
name_tag mixed_bern_K$((K))L$((L))F$((F)) &

wait


K=10
L=4
F=1
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/peptides/peptides_func-GSE-Poly.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.full.repeats $((F)) \
name_tag mixed_bern_K$((K))L$((L))F$((F)) &

wait
