
CUDA_VISIBLE_DEVICES=3 \
python main.py --repeat 5 --cfg configs/GSE/pattern/pattern-GSE_dense-Poly.yaml \
optim.base_lr 0.001 gse_model.attn_drop_prob 0.25 dataset.on_the_fly True &

wait

CUDA_VISIBLE_DEVICES=3 \
python main.py --repeat 5 --cfg configs/GSE/pattern/pattern-GSE_grit-Poly.yaml \
optim.base_lr 0.001 gse_model.attn_drop_prob 0.25 dataset.on_the_fly True &

wait
