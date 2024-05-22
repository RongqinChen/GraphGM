CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/mnist/mnist-GSE_grit-Poly-sparse.yaml  &

wait


CUDA_VISIBLE_DEVICES=0 \
python main.py --repeat 5 --cfg configs/GSE/mnist/mnist-GSE_dense-Poly-sparse.yaml  &

wait
