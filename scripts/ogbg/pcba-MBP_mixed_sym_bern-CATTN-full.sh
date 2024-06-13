for((seed=4;seed>=0;seed--));  
do   

K=14
python main.py \
--cfg configs/MBP/ogbg/pcba-MBP_mixed_sym_bern-CATTN-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
dataset.precompute_on_the_fly True \
name_tag K$((K)) &

wait

done
