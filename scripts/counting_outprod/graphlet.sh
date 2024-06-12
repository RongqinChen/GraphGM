
for((task=0;task<5;task++));  
do

CUDA_VISIBLE_DEVICES=0 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count_outerprod/count-MBP_adj_powers-CATTN-full.yaml --task $((task)) --name_tag outerprodGraphlet$((task))  &

CUDA_VISIBLE_DEVICES=1 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count_outerprod/count-MBP-mixed_low_bern-CATTN-full.yaml --task $((task)) --name_tag outerprodGraphlet$((task))  &

CUDA_VISIBLE_DEVICES=2 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count_outerprod/count-MBP-mixed_sym_bern-CATTN-full.yaml --task $((task)) --name_tag outerprodGraphlet$((task))  &

wait

done  


for((task=0;task<5;task++));  
do

CUDA_VISIBLE_DEVICES=0 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count_outerprod/count-MBP_adj_powers-GINE-full.yaml --task $((task)) --name_tag outerprodGraphlet$((task))  &

CUDA_VISIBLE_DEVICES=1 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count_outerprod/count-MBP-mixed_low_bern-GINE-full.yaml --task $((task)) --name_tag outerprodGraphlet$((task))  &

CUDA_VISIBLE_DEVICES=2 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count_outerprod/count-MBP-mixed_sym_bern-GINE-full.yaml --task $((task)) --name_tag outerprodGraphlet$((task))  &

wait

done  

