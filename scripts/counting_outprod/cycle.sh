for((task=0;task<4;task++));  
do

CUDA_VISIBLE_DEVICES=0 python train_count.py --config_file configs/MBP/count_outerprod/count-MBP_adj_powers-GINE-full.yaml --task $((task)) --name_tag outerprodCircle$((task))  &

CUDA_VISIBLE_DEVICES=1 python train_count.py --config_file configs/MBP/count_outerprod/count-MBP-mixed_low_bern-GINE-full.yaml --task $((task)) --name_tag outerprodCircle$((task))  &

CUDA_VISIBLE_DEVICES=2 python train_count.py --config_file configs/MBP/count_outerprod/count-MBP-mixed_sym_bern-GINE-full.yaml --task $((task)) --name_tag outerprodCircle$((task))  &

wait

done

for((task=0;task<4;task++));
do

CUDA_VISIBLE_DEVICES=0 python train_count.py --config_file configs/MBP/count_outerprod/count-MBP_adj_powers-CATTN-full.yaml --task $((task)) --name_tag outerprodCircle$((task))  &

CUDA_VISIBLE_DEVICES=1 python train_count.py --config_file configs/MBP/count_outerprod/count-MBP-mixed_low_bern-CATTN-full.yaml --task $((task)) --name_tag outerprodCircle$((task))  &

CUDA_VISIBLE_DEVICES=2 python train_count.py --config_file configs/MBP/count_outerprod/count-MBP-mixed_sym_bern-CATTN-full.yaml --task $((task)) --name_tag outerprodCircle$((task))  &

wait

done  
