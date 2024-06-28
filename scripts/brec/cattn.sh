CUDA_VISIBLE_DEVICES=0 python train_BREC.py \
    --config_file configs/MBP/brec/brec-MBP_adj_powers-CATTN-full.yaml &

CUDA_VISIBLE_DEVICES=0 python train_BREC.py \
    --config_file configs/MBP/brec/brec-MBP_mixed_low_bern-CATTN-full.yaml &

CUDA_VISIBLE_DEVICES=0 python train_BREC.py \
    --config_file configs/MBP/brec/brec-MBP_mixed_sym_bern-CATTN-full.yaml &

CUDA_VISIBLE_DEVICES=0 python train_BREC.py \
    --config_file configs/MBP/brec/brec-MBP_spect_adj_powers-CATTN-full.yaml &

wait
