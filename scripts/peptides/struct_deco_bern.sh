


for((seed=0;seed<5;seed++));  
do   

K=8
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/DecoNet/struct-deco_bern.yaml \
seed $((seed)) \
posenc_Poly.power $((K)) \
DecoNet.drop_prob 0.00 \
DecoNet.attn_drop_prob 0.20 \
DecoNet.conv.num_blocks 4 \
name_tag K$((K))DP00ADP20

done
