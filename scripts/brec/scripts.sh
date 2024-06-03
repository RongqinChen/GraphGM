python train_BREC.py --config_file configs/MBP/brec/brec-MBP_mixedbern-GRIT-full.yaml &
wait

python train_BREC.py --config_file configs/MBP/brec/brec-MBP_mixedbern-GRIT-sparse.yaml &
wait

python train_BREC.py --config_file configs/MBP/brec/brec-MBP_adj_powers-GRIT-full.yaml &
wait

python train_BREC.py --config_file configs/MBP/brec/brec-MBP_adj_powers-GRIT-sparse.yaml &
wait
