
CUDA_VISIBLE_DEVICES=3 \
python main.py --repeat 5 --cfg configs/GSE/cluster/cluster-GSE_dense-Poly.yaml  &

wait

CUDA_VISIBLE_DEVICES=3 \
python main.py --repeat 5 --cfg configs/GSE/cluster/cluster-GSE_grit-Poly.yaml  &

wait
