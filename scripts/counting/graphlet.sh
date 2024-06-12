
for((task=0;task<5;task++));
do

CUDA_VISIBLE_DEVICES=0 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count/count-MBP_adj_powers-CATTN-full.yaml --task $((task)) --name_tag Graphlet$((task))  &

CUDA_VISIBLE_DEVICES=1 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count/count-MBP_bern-CATTN-full.yaml --task $((task)) --name_tag Graphlet$((task))  &

CUDA_VISIBLE_DEVICES=2 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count/count-MBP-mixed_sym_bern-CATTN-full.yaml --task $((task)) --name_tag Graphlet$((task))  &

CUDA_VISIBLE_DEVICES=3 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count/count-MBP-mixed_low_bern-CATTN-full.yaml --task $((task)) --name_tag Graphlet$((task))  &

wait
done  


for((task=0;task<5;task++));  
do

CUDA_VISIBLE_DEVICES=0 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count/count-MBP_adj_powers-GINE-full.yaml --task $((task)) --name_tag Graphlet$((task))  &

CUDA_VISIBLE_DEVICES=1 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count/count-MBP-bern-GINE-full.yaml --task $((task)) --name_tag Graphlet$((task))  &

CUDA_VISIBLE_DEVICES=2 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count/count-MBP-mixed_sym_bern-GINE-full.yaml --task $((task)) --name_tag Graphlet$((task))  &

CUDA_VISIBLE_DEVICES=3 python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count/count-MBP-mixed_low_bern-GINE-full.yaml --task $((task)) --name_tag Graphlet$((task))  &

wait
done
