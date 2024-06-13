K=15
seed=4
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/zinc/zinc-MBP_adj_powers.yaml \
seed $((seed)) \
posenc_Poly.power $((K)) \
mbp_model.attn_drop_prob 0.20 \
mbp_model.messaging.num_blocks 0 \
mbp_model.full.repeats 10 \
name_tag K$((K))ADP20 



for((seed=0;seed<2;seed++));  
do   

K=15
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/zinc/zinc-MBP_adj_powers.yaml \
seed $((2*seed)) \
posenc_Poly.power $((K)) \
mbp_model.attn_drop_prob 0.20 \
mbp_model.messaging.num_blocks 0 \
mbp_model.full.repeats 10 \
name_tag K$((K))ADP20 &

# wait

K=15
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/zinc/zinc-MBP_adj_powers.yaml \
seed $((2*seed+1)) \
posenc_Poly.power $((K)) \
mbp_model.attn_drop_prob 0.20 \
mbp_model.messaging.num_blocks 0 \
mbp_model.full.repeats 10 \
name_tag K$((K))ADP20 &

wait

done  
