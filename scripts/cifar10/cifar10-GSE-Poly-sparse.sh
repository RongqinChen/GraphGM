
CUDA_VISIBLE_DEVICES=1 \
python main.py --repeat 5 --cfg configs/GSE/cifar/cifar10-GSE_grit-Poly-sparse.yaml  &

wait

CUDA_VISIBLE_DEVICES=1 \
python main.py --repeat 5 --cfg configs/GSE/cifar/cifar10-GSE_dense-Poly-sparse.yaml  &

wait
