for((seed=4;seed>=0;seed--));  
do   

K=8 \
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/GSE/zinc/zincfull-GSE.yaml seed $((seed))  \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.messaging.num_blocks 4 \
gse_model.messaging.repeats 2 \
gse_model.full.repeats 2 \
name_tag LowMiddleK$((K)) &

wait

K=16 \
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/GSE/zinc/zincfull-GSE.yaml seed $((seed))  \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.hidden_dim 56 \
gse_model.messaging.num_blocks 5 \
gse_model.messaging.repeats 2 \
gse_model.full.repeats 2 \
name_tag LowMiddleK$((K)) &

wait

done  

