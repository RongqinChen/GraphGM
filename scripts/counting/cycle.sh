for((task=0;task<=2;task++));  
do

CUDA_VISIBLE_DEVICES=0 python train_count.py --config_file configs/MBP/count/count-MBP_adj_powers-GINE-full.yaml --task $((task)) --name_tag Circle$((task))  &

CUDA_VISIBLE_DEVICES=1 python train_count.py --config_file configs/MBP/count/count-MBP_bern-GINE-full.yaml --task $((task)) --name_tag Circle$((task))  &

CUDA_VISIBLE_DEVICES=2 python train_count.py --config_file configs/MBP/count/count-MBP-mixed_low_bern-GINE-full.yaml --task $((task)) --name_tag Circle$((task))  &

CUDA_VISIBLE_DEVICES=3 python train_count.py --config_file configs/MBP/count/count-MBP-mixed_sym_bern-GINE-full.yaml --task $((task)) --name_tag Circle$((task))  &

wait

done  
