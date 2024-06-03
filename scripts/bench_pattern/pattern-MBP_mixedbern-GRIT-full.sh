for((seed=4;seed>=0;seed--));  
do   

K=4 
CUDA_VISIBLE_DEVICES=3 \
python main.py --cfg configs/MBP/bench_pattern/pattern-MBP_mixedbern-GRIT-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
name_tag K$((K)) &

wait

K=6 
CUDA_VISIBLE_DEVICES=3 \
python main.py --cfg configs/MBP/bench_pattern/pattern-MBP_mixedbern-GRIT-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
name_tag K$((K)) &

wait

K=8 
CUDA_VISIBLE_DEVICES=3 \
python main.py --cfg configs/MBP/bench_pattern/pattern-MBP_mixedbern-GRIT-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
name_tag K$((K)) &

wait

done  
