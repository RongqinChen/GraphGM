K=10
CUDA_VISIBLE_DEVICES=0 \
    python main.py --repeat 5 --cfg configs/GSE/contact/pcqm_contact-GRIT-MixedBern.yaml  \
    posenc_Poly.method mixed_bern posenc_Poly.order $((K)) posenc_Poly.emb_dim $(( (K+2) ))  \
    name_tag mixed_bern_K$((K)) &

wait
