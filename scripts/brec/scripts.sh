CUDA_VISIBLE_DEVICES=0 \
python train_BREC.py --config_file configs/MBP/brec/brec-MBP_adj_powers-GRIT-full.yaml &

CUDA_VISIBLE_DEVICES=1 \
python train_BREC.py --config_file configs/MBP/brec/brec-MBP_adj_powers-GRIT-sparse.yaml &

wait
