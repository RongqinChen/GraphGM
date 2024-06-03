for((seed=4;seed>=0;seed--));  
do   

K=4 
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/contact/pcqm_contact-MBP_mixedbern-GRIT-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
name_tag K$((K)) &

wait

K=6 
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/contact/pcqm_contact-MBP_mixedbern-GRIT-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
name_tag K$((K)) &

wait

K=8 
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/contact/pcqm_contact-MBP_mixedbern-GRIT-full.yaml seed $((seed))  \
posenc_Poly.power $((K)) \
name_tag K$((K)) &

wait

done  
