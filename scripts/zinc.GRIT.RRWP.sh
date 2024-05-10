
# CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/zinc-GRIT-RRWP.yaml wandb.use False seed 0  &
# CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/zinc-GRIT-RRWP.yaml wandb.use False seed 1  &
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/zinc-GRIT-RRWP.yaml wandb.use False seed 2  &
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/zinc-GRIT-RRWP.yaml wandb.use False seed 3  &
wait

python -m graphgps.agg_runs 
