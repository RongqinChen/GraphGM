

for((seed=0;seed<5;seed++));  
do   

K=8
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/zinc/zinc-MBP_spect_adj_powers-CATTN-full.yaml \
seed $((seed)) \
posenc_Poly.power $((K)) \
name_tag K$((K))ADP20 

# mbp_model.attn_drop_prob 0.20 \
# mbp_model.messaging.num_blocks 0 \
# mbp_model.full.repeats 10 \

done  


for((seed=0;seed<5;seed++));  
do   

K=10
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/zinc/zinc-MBP_spect_adj_powers-CATTN-full.yaml \
seed $((seed)) \
posenc_Poly.power $((K)) \
name_tag K$((K))ADP20 

# mbp_model.attn_drop_prob 0.20 \
# mbp_model.messaging.num_blocks 0 \
# mbp_model.full.repeats 10 \

done  


# for((seed=0;seed<5;seed++));  
# do   

# K=12
# CUDA_VISIBLE_DEVICES=0 python main.py \
# --cfg configs/MBP/zinc/zinc-MBP_spect_adj_powers-CATTN-full.yaml \
# seed $((seed)) \
# posenc_Poly.power $((K)) \
# name_tag K$((K))ADP20 

# # mbp_model.attn_drop_prob 0.20 \
# # mbp_model.messaging.num_blocks 0 \
# # mbp_model.full.repeats 10 \

# done  


for((seed=0;seed<5;seed++));  
do   

K=14
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/zinc/zinc-MBP_spect_adj_powers-CATTN-full.yaml \
seed $((seed)) \
posenc_Poly.power $((K)) \
name_tag K$((K))ADP20 

# mbp_model.attn_drop_prob 0.20 \
# mbp_model.messaging.num_blocks 0 \
# mbp_model.full.repeats 10 \

done  
