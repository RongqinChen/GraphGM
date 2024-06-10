for((seed=4;seed>=0;seed--));  
do   

K=18 
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/bench_cifar10/cifar10-MBP_mixed_low_bern-GRIT-full.yaml \
seed $((seed))  \
posenc_Poly.power $((K)) \
name_tag K$((K))

done  
