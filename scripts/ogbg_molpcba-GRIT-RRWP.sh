
CUDA_VISIBLE_DEVICES=0 python main.py --repeat 5 --cfg configs/GRIT/ogbg_molpcba-GRIT-RRWP.yaml print stdout

python -m graphgps.agg_runs 
