for((task=3;task>=0;task--));  
do

python train_count.py --config_file configs/MBP/count/count-MBP_adj_powers-GRIT-full.yaml --task $((task)) --name_tag T$((task))K24L10Dp20  &

python train_count.py --config_file configs/MBP/count/count-MBP_adj_powers-GRIT-sparse.yaml --task $((task)) --name_tag T$((task))K24L10Dp20  &

python train_count.py --config_file configs/MBP/count/count-MBP_mixedbern-GRIT-full.yaml --task $((task)) --name_tag T$((task))K24L10Dp20  &

python train_count.py --config_file configs/MBP/count/count-MBP_mixedbern-GRIT-sparse.yaml --task $((task)) --name_tag T$((task))K24L10Dp20  &

wait

done  
