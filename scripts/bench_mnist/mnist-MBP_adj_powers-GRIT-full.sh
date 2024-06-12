for((seed=4;seed>=0;seed--));  
do   

K=18 
CUDA_VISIBLE_DEVICES=2 python main.py \
--cfg configs/MBP/bench_mnist/mnist-MBP_adj_powers-CATTN-full.yaml \
seed $((seed))  \
posenc_Poly.power $((K)) \
name_tag K$((K))

done  
