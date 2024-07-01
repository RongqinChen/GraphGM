for((seed=4;seed>=0;seed--));  
do

K=8
python main.py --cfg configs/MBP/peptides/struct-MBP_mixed_sym_bern-CATTN-full.yaml seed $((seed)) name_tag K$((K)) 

K=6
python main.py --cfg configs/MBP/peptides/struct-MBP_mixed_sym_bern-CATTN-full.yaml seed $((seed)) name_tag K$((K))

K=10
python main.py --cfg configs/MBP/peptides/struct-MBP_mixed_sym_bern-CATTN-full.yaml seed $((seed)) name_tag K$((K)) 

done 
