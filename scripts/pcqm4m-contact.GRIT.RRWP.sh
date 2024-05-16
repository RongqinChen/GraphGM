
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/pcqm_contact-GRIT-RRWP.yaml wandb.use False seed 0 print stdout &
wait

CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/pcqm_contact-GRIT-RRWP.yaml wandb.use False seed 1 print stdout &
wait

CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/pcqm_contact-GRIT-RRWP.yaml wandb.use False seed 2 print stdout &
wait

CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/pcqm_contact-GRIT-RRWP.yaml wandb.use False seed 3 print stdout &
wait

CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GRIT/pcqm_contact-GRIT-RRWP.yaml wandb.use False seed 4 print stdout &
wait

python -m graphgps.agg_runs 
