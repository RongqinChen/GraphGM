CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/mnist-GRIT-Bern.yaml wandb.use False seed 0  &
wait

CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/mnist-GRIT-Bern.yaml wandb.use False seed 1  &
wait

CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/mnist-GRIT-Bern.yaml wandb.use False seed 2  &
wait

CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/mnist-GRIT-Bern.yaml wandb.use False seed 3  &
wait

python -m graphgps.agg_runs 
