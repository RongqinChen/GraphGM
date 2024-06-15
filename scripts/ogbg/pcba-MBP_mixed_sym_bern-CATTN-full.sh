# for((seed=4;seed>=0;seed--));  
# do   

# K=14
# python main.py \
# --cfg configs/MBP/ogbg/pcba-MBP_mixed_sym_bern-CATTN-full.yaml seed $((seed))  \
# posenc_Poly.power $((K)) \
# dataset.precompute_on_the_fly True \
# name_tag K$((K)) &

# wait

# done



K=14

seed=0 
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/ogbg/pcba-MBP_mixed_sym_bern-CATTN-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
dataset.precompute_on_the_fly True \
name_tag K$((K))  &


seed=1 
CUDA_VISIBLE_DEVICES=1 python main.py \
--cfg configs/MBP/ogbg/pcba-MBP_mixed_sym_bern-CATTN-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
dataset.precompute_on_the_fly True \
name_tag K$((K))  &

wait

seed=2 
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/ogbg/pcba-MBP_mixed_sym_bern-CATTN-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
dataset.precompute_on_the_fly True \
name_tag K$((K))  &


seed=3 
CUDA_VISIBLE_DEVICES=1 python main.py \
--cfg configs/MBP/ogbg/pcba-MBP_mixed_sym_bern-CATTN-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
dataset.precompute_on_the_fly True \
name_tag K$((K))  &

wait

seed=4 
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg configs/MBP/ogbg/pcba-MBP_mixed_sym_bern-CATTN-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
dataset.precompute_on_the_fly True \
name_tag K$((K))  &

wait
