
CUDA_VISIBLE_DEVICES=3 \
python main.py --repeat 5 --cfg configs/GSE/mnist/mnist-GSE_grit-Poly.yaml  &

wait

CUDA_VISIBLE_DEVICES=3 \
python main.py --repeat 5 --cfg configs/GSE/mnist/mnist-GSE_dense-Poly.yaml  &

wait
