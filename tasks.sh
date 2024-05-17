# K=14
# CUDA_VISIBLE_DEVICES=0 \
#     python main.py --repeat 5 --cfg configs/GSE/zinc/zinc-GT-GRIT-Poly.yaml \
#     posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2)*(K+1)/2 )) name_tag K$((K)) &
# wait

K=24
CUDA_VISIBLE_DEVICES=0 \
    python main.py --repeat 5 --cfg configs/GSE/zinc/zinc-GT-GRIT-Poly.yaml  \
    posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
    name_tag mixed_bern_K$((K)) &

wait

K=28
CUDA_VISIBLE_DEVICES=1 \
    python main.py --repeat 5 --cfg configs/GSE/zinc/zinc-GT-GRIT-Poly.yaml  \
    posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
    name_tag mixed_bern_K$((K)) &

wait

# K=14
# CUDA_VISIBLE_DEVICES=0 \
#     python main.py --repeat 5 --cfg configs/GSE/zinc/zinc-GT-Add-Poly.yaml \
#     posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2)*(K+1)/2 )) name_tag K$((K)) &
# wait

# K=14
# CUDA_VISIBLE_DEVICES=0 \
#     python main.py --repeat 5 --cfg configs/GSE/pattern/pattern-GT-GRIT-Poly.yaml \
#     posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2)*(K+1)/2 )) name_tag K$((K)) &
# wait


# K=14
# CUDA_VISIBLE_DEVICES=0 \
#     python main.py --repeat 5 --cfg configs/GSE/pattern/pattern-GT-Add-Poly.yaml \
#     posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2)*(K+1)/2 )) name_tag K$((K)) &
# wait

