
for((seed=0;seed<5;seed++));  
do   

K=8
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/zinc/zinc-MBP_adj_powers.yaml \
seed $((seed)) \
posenc_Poly.power $((K)) \
name_tag K$((K))ADP20 

done  


for((seed=0;seed<5;seed++));  
do   

K=10
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/zinc/zinc-MBP_adj_powers.yaml \
seed $((seed)) \
posenc_Poly.power $((K)) \
name_tag K$((K))ADP20 

done  



for((seed=0;seed<5;seed++));  
do   

K=12
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/zinc/zinc-MBP_adj_powers.yaml \
seed $((seed)) \
posenc_Poly.power $((K)) \
name_tag K$((K))ADP20 

done  
