for((seed=4;seed>=0;seed--));  
do


K=8
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/peptides/peptides_struct-MBP_mixedbern-GRIT-sparse.yaml \
posenc_Poly.power $((K)) mbp_model.drop_prob 0.0 \
mbp_model.hidden_dim 80 mbp_model.messaging.num_blocks 4 \
seed $((seed)) name_tag K$((K))DP00 &

# wait

K=8
CUDA_VISIBLE_DEVICES=1 \
python main.py --cfg configs/MBP/peptides/peptides_struct-MBP_mixedbern-GRIT-sparse.yaml \
posenc_Poly.power $((K)) mbp_model.drop_prob 0.05 \
mbp_model.hidden_dim 80 mbp_model.messaging.num_blocks 4 \
seed $((seed)) name_tag K$((K))DP05 &

wait

# K=12
# CUDA_VISIBLE_DEVICES=0 \
# python main.py --cfg configs/MBP/peptides/peptides_struct-MBP_mixedbern-GRIT-sparse.yaml \
# posenc_Poly.power $((K)) mbp_model.hidden_dim 80 mbp_model.messaging.num_blocks 5 \
# seed $((seed)) name_tag K$((K)) &

# wait

done  
