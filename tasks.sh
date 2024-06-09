
K=8
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/zinc/zinc-MBP_bern-CATTN-full.yaml seed 4  \
posenc_Poly.power $((K)) \
mbp_model.attn_drop_prob 0.20 \
mbp_model.drop_prob 0.00 \
mbp_model.messaging.num_blocks 0 \
mbp_model.full.repeats 10 \
name_tag K$((K))ADP20DP00 &

# wait


K=8
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/zinc/zinc-MBP_bern-CATTN-full.yaml seed 3  \
posenc_Poly.power $((K)) \
mbp_model.attn_drop_prob 0.20 \
mbp_model.drop_prob 0.00 \
mbp_model.messaging.num_blocks 0 \
mbp_model.full.repeats 10 \
name_tag K$((K))ADP20DP00 &

wait

K=8
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/zinc/zinc-MBP_bern-CATTN-full.yaml seed 2  \
posenc_Poly.power $((K)) \
mbp_model.attn_drop_prob 0.20 \
mbp_model.drop_prob 0.00 \
mbp_model.messaging.num_blocks 0 \
mbp_model.full.repeats 10 \
name_tag K$((K))ADP20DP00 &

# wait

K=8
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/zinc/zinc-MBP_bern-CATTN-full.yaml seed 1  \
posenc_Poly.power $((K)) \
mbp_model.attn_drop_prob 0.20 \
mbp_model.drop_prob 0.00 \
mbp_model.messaging.num_blocks 0 \
mbp_model.full.repeats 10 \
name_tag K$((K))ADP20DP00 &

# wait

K=8
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/zinc/zinc-MBP_bern-CATTN-full.yaml seed 0  \
posenc_Poly.power $((K)) \
mbp_model.attn_drop_prob 0.20 \
mbp_model.drop_prob 0.00 \
mbp_model.messaging.num_blocks 0 \
mbp_model.full.repeats 10 \
name_tag K$((K))ADP20DP00 &

wait
