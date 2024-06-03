
K=8 
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/zinc/zinc-MBP_mixedbern-GRIT-poly.yaml seed 0  \
posenc_Poly.power $((K)) \
mbp_model.drop_prob 0.01 \
mbp_model.attn_drop_prob 0.20 \
name_tag K$((K))ADP20DP01 &

# wait


K=8 
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/zinc/zinc-MBP_mixedbern-GRIT-poly.yaml seed 1  \
posenc_Poly.power $((K)) \
mbp_model.drop_prob 0.01 \
mbp_model.attn_drop_prob 0.20 \
name_tag K$((K))ADP20DP01 &

wait


K=8 
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/zinc/zinc-MBP_mixedbern-GRIT-poly.yaml seed 2  \
posenc_Poly.power $((K)) \
mbp_model.drop_prob 0.01 \
mbp_model.attn_drop_prob 0.20 \
name_tag K$((K))ADP20DP01 &

# wait


K=8 
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/zinc/zinc-MBP_mixedbern-GRIT-poly.yaml seed 3  \
posenc_Poly.power $((K)) \
mbp_model.drop_prob 0.01 \
mbp_model.attn_drop_prob 0.20 \
name_tag K$((K))ADP20DP01 &

wait


K=8 
CUDA_VISIBLE_DEVICES=0 \
python main.py --cfg configs/MBP/zinc/zinc-MBP_mixedbern-GRIT-poly.yaml seed 4  \
posenc_Poly.power $((K)) \
mbp_model.drop_prob 0.01 \
mbp_model.attn_drop_prob 0.20 \
name_tag K$((K))ADP20DP01 &

wait
