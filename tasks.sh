K=14
CUDA_VISIBLE_DEVICES=0 \
    python main.py --repeat 5 --cfg configs/GSE/zinc/GRIT-RRW_Bern.yaml \
    posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2)*(K+1)/2 )) name_tag K$((K)) &
wait


K=14
CUDA_VISIBLE_DEVICES=0  \
    python main.py --repeat 5 --cfg configs/GSE/pattern/pattern-RRW_Bern.yaml \
    posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2)*(K+1)/2 )) name_tag K$((K)) &
wait
