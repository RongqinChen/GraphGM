CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/zinc-GRIT-RRW_Bern.yaml wandb.use False seed 0 name_tag T102038  &
# wait

CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/GRIT/zinc-GRIT-RRW_Bern.yaml wandb.use False seed 1 name_tag T102038  &
wait

CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/zinc-GRIT-RRW_Bern.yaml wandb.use False seed 2 name_tag T102038  &
# wait

CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/GRIT/zinc-GRIT-RRW_Bern.yaml wandb.use False seed 3 name_tag T102038  &
wait

CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/zinc-GRIT-RRW_Bern.yaml wandb.use False seed 4 name_tag T102038  &
wait

python -m graphgps.agg_runs 
