for((seed=4;seed>=0;seed--));  
do   

K=32
CUDA_VISIBLE_DEVICES=1 python main.py \
--cfg configs/MBP/bench_cluster/cluster-MBP_mixed_low_bern-CATTN-full.yaml 
seed $((seed))  \
posenc_Poly.power $((K)) \
name_tag K$((K)) 

done  
