
K=8
CUDA_VISIBLE_DEVICES=0 \
    python main.py --cfg configs/GSE/zinc/zinc-GT-Add-Poly.yaml  \
    posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
    name_tag mixed_bern_K$((K)) seed 4 &

wait


K=6
CUDA_VISIBLE_DEVICES=0 \
    python main.py --cfg configs/GSE/zinc/zinc-GT-Add-Poly.yaml  \
    posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
    name_tag mixed_bern_K$((K)) seed 4 &

wait


K=10
CUDA_VISIBLE_DEVICES=0 \
    python main.py --cfg configs/GSE/zinc/zinc-GT-Add-Poly.yaml  \
    posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
    name_tag mixed_bern_K$((K)) seed 4 &

wait
