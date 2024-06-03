for((task=4;task>=0;task--));  
do

python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count/count-MBP_adj_powers-GRIT-full.yaml --task $((task)) --name_tag G$((task))K24L10Dp20  &

python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count/count-MBP_adj_powers-GRIT-sparse.yaml --task $((task)) --name_tag G$((task))K24L10Dp20  &

python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count/count-MBP_mixedbern-GRIT-full.yaml --task $((task)) --name_tag G$((task))K24L10Dp20  &

python train_count.py --dataset_name count_graphlet --config_file configs/MBP/count/count-MBP_mixedbern-GRIT-sparse.yaml --task $((task)) --name_tag G$((task))K24L10Dp20  &

wait

done  
