
CUDA_VISIBLE_DEVICES=1 \
python main.py --repeat 5 --cfg configs/GSE/pattern/pattern-GSE_dense-Poly.yaml \
optim.base_lr 0.0005 gse_model.attn_drop_prob 0.2 dataset.on_the_fly False &

wait

CUDA_VISIBLE_DEVICES=1 \
python main.py --repeat 5 --cfg configs/GSE/pattern/pattern-GSE_grit-Poly.yaml \
optim.base_lr 0.0005 gse_model.attn_drop_prob 0.2 dataset.on_the_fly False &

wait
