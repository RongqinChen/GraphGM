
which python

source /home/yc07917/mambaforge/bin/activate gnn210
which python


for((seed=4;seed>=0;seed--));  
do   

K=12
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/GSE/peptides/peptides_struct-GSE_grit-Poly-full.yaml seed $((seed))  \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.attn_drop_prob 0.2 \
gse_model.messaging.num_blocks 0 \
gse_model.full.repeats 4 \
name_tag LowMiddleK$((K))Adp20 &

wait


K=10
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/GSE/peptides/peptides_struct-GSE_grit-Poly-full.yaml seed $((seed))  \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.attn_drop_prob 0.2 \
gse_model.messaging.num_blocks 0 \
gse_model.full.repeats 4 \
name_tag LowMiddleK$((K))Adp20 & 

wait


K=8
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/GSE/peptides/peptides_struct-GSE_grit-Poly-full.yaml seed $((seed))  \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.attn_drop_prob 0.2 \
gse_model.messaging.num_blocks 0 \
gse_model.full.repeats 4 \
name_tag LowMiddleK$((K))Adp20 & 

wait


K=14
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/GSE/peptides/peptides_struct-GSE_grit-Poly-full.yaml seed $((seed))  \
posenc_Poly.method low_middle_pass posenc_Poly.order $((K)) posenc_Poly.emb_dim $((K))  \
gse_model.attn_drop_prob 0.2 \
gse_model.messaging.num_blocks 0 \
gse_model.full.repeats 4 \
name_tag LowMiddleK$((K))Adp20 & 

wait

done  
