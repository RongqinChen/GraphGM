
which python

source /home/yc07917/mambaforge/bin/activate gnn210
which python


for((seed=4;seed>=0;seed--));  
do   

K=8
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/peptides/peptides_struct-MBP_mixedbern-GRIT-full.yaml  seed $((seed)) &

wait

K=16
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/peptides/peptides_struct-MBP_mixedbern-GRIT-full.yaml  seed $((seed)) &

wait

K=10
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/peptides/peptides_struct-MBP_mixedbern-GRIT-full.yaml  seed $((seed)) &

wait

K=12
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/peptides/peptides_struct-MBP_mixedbern-GRIT-full.yaml  seed $((seed)) &

wait

done  
