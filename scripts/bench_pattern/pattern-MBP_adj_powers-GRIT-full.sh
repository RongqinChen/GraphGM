for((seed=4;seed>=0;seed--));  
do   

K=20
CUDA_VISIBLE_DEVICES=3 \
python main.py --cfg configs/MBP/bench_pattern/pattern-MBP_adj_powers-CATTN-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
name_tag K$((K))

done  
