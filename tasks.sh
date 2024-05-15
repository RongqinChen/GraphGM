CUDA_VISIBLE_DEVICES=0 K=12 \
    python main.py --cfg configs/zinc/GRIT-RRW_Bern.yaml \
    posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2)*(K+1)/2 )) name_tag K$((K)) &
wait

