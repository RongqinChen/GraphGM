
CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/cluster/cluster-GSE_grit-Poly.yaml \
optim.base_lr 0.0005 gse_model.attn_drop_prob 0.5 dataset.on_the_fly False &

wait

CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/cluster/cluster-GSE_dense-Poly.yaml \
optim.base_lr 0.0005 gse_model.attn_drop_prob 0.5 dataset.on_the_fly False &

wait
