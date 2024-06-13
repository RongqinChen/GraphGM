# export CUDA_VISIBLE_DEVICES=1
CUDA_VISIBLE_DEVICES=0 python train_BREC.py --config_file configs/MBP/brec/brec-MBP_adj_powers-CATTN-full.yaml &
# wait

# python train_BREC.py --config_file configs/MBP/brec/brec-MBP_bern-CATTN-full.yaml &
# wait

CUDA_VISIBLE_DEVICES=1 python train_BREC.py --config_file configs/MBP/brec/brec-MBP_mixed_sym_bern-CATTN-full.yaml &
wait

# python train_BREC.py --config_file configs/MBP/brec/brec-MBP_mixed_low_bern-CATTN-full.yaml &
# wait
