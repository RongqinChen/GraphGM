
K=16
R=1
L=5
F=1
H=72
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/peptides/peptides_func-GSE_grit-Poly.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.messaging.repeats $((R)) \
gse_model.full.repeats $((F)) \
gse_model.hidden_dim $((H)) \
dataset.on_the_fly True \
name_tag mixed_bern_K$((K))R$((R))L$((L))F$((F)) 



K=8
R=1
L=4
F=1
H=80
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/peptides/peptides_func-GSE_grit-Poly.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.messaging.repeats $((R)) \
gse_model.full.repeats $((F)) \
gse_model.hidden_dim $((H)) \
dataset.on_the_fly True \
name_tag mixed_bern_K$((K))R$((R))L$((L))F$((F)) 


K=4
R=1
L=3
F=1
H=88
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/peptides/peptides_func-GSE_grit-Poly.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.messaging.repeats $((R)) \
gse_model.full.repeats $((F)) \
gse_model.hidden_dim $((H)) \
dataset.on_the_fly False \
name_tag mixed_bern_K$((K))R$((R))L$((L))F$((F))


K=16
R=1
L=5
F=1
H=72
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/peptides/peptides_func-GSE_dense-Poly.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.messaging.repeats $((R)) \
gse_model.full.repeats $((F)) \
gse_model.hidden_dim $((H)) \
dataset.on_the_fly True \
name_tag mixed_bern_K$((K))R$((R))L$((L))F$((F)) 



K=8
R=1
L=4
F=1
H=80
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/peptides/peptides_func-GSE_dense-Poly.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.messaging.repeats $((R)) \
gse_model.full.repeats $((F)) \
gse_model.hidden_dim $((H)) \
dataset.on_the_fly True \
name_tag mixed_bern_K$((K))R$((R))L$((L))F$((F)) 


K=4
R=1
L=3
F=1
H=88
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/peptides/peptides_func-GSE_dense-Poly.yaml  \
posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
gse_model.messaging.num_blocks $((L)) \
gse_model.messaging.repeats $((R)) \
gse_model.full.repeats $((F)) \
gse_model.hidden_dim $((H)) \
dataset.on_the_fly False \
name_tag mixed_bern_K$((K))R$((R))L$((L))F$((F))

