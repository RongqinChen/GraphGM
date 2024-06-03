
for((seed=4;seed>=0;seed--));  
do   

K=8 
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/pcqm4mv2/pcqm4mv2-MBP_mixedbern-GRIT-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
name_tag K$((K))

done  
