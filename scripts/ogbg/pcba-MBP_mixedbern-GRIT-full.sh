for((seed=4;seed>=0;seed--));  
do   

# K=8
# L=5
# CUDA_VISIBLE_DEVICES=0 \
# python main.py --cfg configs/MBP/ogbg/pcba-MBP_mixedbern-GRIT-full.yaml  \
# posenc_Poly.method mixed_bern posenc_Poly.power $((K)) \
# mbp_model.hidden_dim 256 \
# mbp_model.messaging.num_blocks 0 \
# mbp_model.messaging.repeats 0 \
# mbp_model.full.repeats $((L)) \
# dataset.precompute_on_the_fly True \
# name_tag mixed_bern_K$((K))F$((L)) &



K=8
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/ogbg/pcba-MBP_mixedbern-GRIT-full.yaml seed $((seed))  \
posenc_Poly.method mixed_bern posenc_Poly.power $((K)) \
mbp_model.messaging.num_blocks 4 \
mbp_model.messaging.repeats 1 \
mbp_model.full.repeats 1 \
dataset.precompute_on_the_fly True \
name_tag mixed_bern_K$((K))M4F1 &

wait

done
