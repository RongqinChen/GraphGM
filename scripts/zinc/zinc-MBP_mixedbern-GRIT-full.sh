for((seed=4;seed>=0;seed--));  
do   

K=4 
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/zinc/zinc-MBP_mixedbern-GRIT-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
mbp_model.attn_drop_prob 0.20 \
mbp_model.messaging.num_blocks 0 \
mbp_model.full.repeats 10 \
name_tag K$((K))ADP20 &

# wait

K=6 
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/zinc/zinc-MBP_mixedbern-GRIT-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
mbp_model.attn_drop_prob 0.20 \
mbp_model.messaging.num_blocks 0 \
mbp_model.full.repeats 10 \
name_tag K$((K))ADP20 &

wait

done  
