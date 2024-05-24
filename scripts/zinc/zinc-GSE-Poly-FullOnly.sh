
K=8
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/zinc/zinc-GSE.yaml  \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.messaging.num_blocks 0 \
gse_model.full.repeats 10 \
name_tag LowMiddleK$((K)) &

# wait

K=12
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/zinc/zinc-GSE.yaml  \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.messaging.num_blocks 0 \
gse_model.full.repeats 10 \
name_tag LowMiddleK$((K)) &

wait

K=6
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/zinc/zinc-GSE.yaml  \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.messaging.num_blocks 0 \
gse_model.full.repeats 10 \
name_tag LowMiddleK$((K)) &

# wait

K=16
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/zinc/zinc-GSE.yaml  \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.messaging.num_blocks 0 \
gse_model.full.repeats 10 \
name_tag LowMiddleK$((K)) &

wait
