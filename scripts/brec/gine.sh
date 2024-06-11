# python train_BREC.py --config_file configs/MBP/brec/brec-MBP_adj_powers-GINE-full.yaml &
# wait

export CUDA_VISIBLE_DEVICES=1

python train_BREC.py --config_file configs/MBP/brec/brec-MBP_bern-GINE-full.yaml &
wait

python train_BREC.py --config_file configs/MBP/brec/brec-MBP_mixed_sym_bern-GINE-full.yaml &
wait
