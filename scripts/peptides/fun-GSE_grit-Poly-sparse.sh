
for((seed=4;seed>=0;seed--));  
do   

CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/GSE/peptides/peptides_func-GSE_grit-Poly-sparse.yaml seed $((seed))  \
gse_model.attn_drop_prob 0.20 \
name_tag LowMiddleAdp20 &

wait


CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/GSE/peptides/peptides_func-GSE_grit-Poly-sparse.yaml seed $((seed))  \
gse_model.attn_drop_prob 0.50 \
name_tag LowMiddleAdp50 &

done
