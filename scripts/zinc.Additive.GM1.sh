# CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/zinc-Additive-GM1.yaml wandb.use False seed 0  &
# wait
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/zinc-Additive-GM1.yaml wandb.use False seed 1  &
# wait
CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/GRIT/zinc-Additive-GM1.yaml wandb.use False seed 2  &
# wait
CUDA_VISIBLE_DEVICES=2 python main.py --cfg configs/GRIT/zinc-Additive-GM1.yaml wandb.use False seed 3  &
# wait
CUDA_VISIBLE_DEVICES=3 python main.py --cfg configs/GRIT/zinc-Additive-GM1.yaml wandb.use False seed 4  &

wait
