python train_BREC.py --config_file configs/MBP/brec/brec-MBP_adj_powers-CATTN-full.yaml &
wait

python train_BREC.py --config_file configs/MBP/brec/brec-MBP_bern-CATTN-full.yaml &
wait

python train_BREC.py --config_file configs/MBP/brec/brec-MBP_mixedbern-CATTN-full.yaml &
wait
