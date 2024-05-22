
CUDA_VISIBLE_DEVICES=1 \
python main.py --repeat 5 --cfg configs/GSE/cifar/cifar10-GSE_grit-Poly.yaml  &

wait

CUDA_VISIBLE_DEVICES=1 \
python main.py --repeat 5 --cfg configs/GSE/cifar/cifar10-GSE_dense-Poly.yaml  &

wait
