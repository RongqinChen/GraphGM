#!/bin/bash

#SBATCH --job-name              pat_grit_bern
#SBATCH --time                  48:00:00
#SBATCH --cpus-per-task         6
#SBATCH --gres                  gpu:1
#SBATCH --mem                   200G
#SBATCH --output=results/pat_grit_bern.out
#SBATCH --partition             h800_batch

your_cleanup_function()
{
    echo "function your_cleanup_function called at $(date)" # do whatever cleanup you want here
    pkill -u rongqin
}

# call your_cleanup_function once we receive USR1 signal 
trap 'your_cleanup_function' USR1


source /home/yc07917/mambaforge/bin/activate gnn210
which python

free -h
echo ""
lscpu
echo ""
nvidia-smi
echo ""

# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/peptides-func-GRIT-Bern.yaml wandb.use False seed 0 
# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/peptides-func-GRIT-Bern.yaml wandb.use False seed 1 
# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/peptides-func-GRIT-Bern.yaml wandb.use False seed 2 
# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/peptides-func-GRIT-Bern.yaml wandb.use False seed 3 
# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/peptides-func-GRIT-Bern.yaml wandb.use False seed 4 

# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/peptides-struct-GRIT-Bern.yaml wandb.use False seed 0 
# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/peptides-struct-GRIT-Bern.yaml wandb.use False seed 1 
# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/peptides-struct-GRIT-Bern.yaml wandb.use False seed 2 
# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/peptides-struct-GRIT-Bern.yaml wandb.use False seed 3 
# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/peptides-struct-GRIT-Bern.yaml wandb.use False seed 4 

# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/cluster-GRIT-Bern.yaml wandb.use False seed 0 
# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/cluster-GRIT-Bern.yaml wandb.use False seed 1 
# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/cluster-GRIT-Bern.yaml wandb.use False seed 2 
# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/cluster-GRIT-Bern.yaml wandb.use False seed 3 
# /home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/cluster-GRIT-Bern.yaml wandb.use False seed 4 

/home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/pattern-GRIT-Bern.yaml wandb.use False seed 0 
/home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/pattern-GRIT-Bern.yaml wandb.use False seed 1 
/home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/pattern-GRIT-Bern.yaml wandb.use False seed 2 
/home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/pattern-GRIT-Bern.yaml wandb.use False seed 3 
/home/yc07917/mambaforge/envs/gnn210/bin/python main.py --cfg configs/GRIT/pattern-GRIT-Bern.yaml wandb.use False seed 4 

echo "finished!!!"
