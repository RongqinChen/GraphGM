for((seed=4;seed>=0;seed--));  
do   

K=8
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/zinc/zincfull-MBP_mixed_sym_bern-CATTN-full.yaml \
seed $((seed)) \
posenc_Poly.power $((K)) \
name_tag K$((K))ADP20

done
