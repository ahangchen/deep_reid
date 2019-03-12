#!/usr/bin/env bash
export PYTHONPATH=/home/cwh/coding/taudl_pyt
export PYTHON=/home/cwh/anaconda3/bin/python
export LD_LIBRAR_PATH=/usr/local/cuda/lib64
#db_names=("grid_label" "viper")
#
#for i in "${!db_names[@]}";
#do
#    data_dir="/home/cwh/coding/${db_names[$i]}/pytorch"
#    # printf $data_dir
#    $PYTHON train.py --gpu_ids 3 --name ${db_names[$i]} --data_dir $data_dir --batchsize 32 --train_all
#done
#
#db_names=("Market" 'DukeMTMC-reID' "cuhk01")
#
#for i in "${!db_names[@]}";
#do
#    data_dir="/home/cwh/coding/${db_names[$i]}/pytorch"
#    # printf $data_dir
#    $PYTHON train.py --gpu_ids 3 --name ${db_names[$i]} --data_dir $data_dir --batchsize 32
#done

for i in {0..9}
do
    data_dir="/home/cwh/coding/dataset/grid-cv-${i}/pytorch"
    $PYTHON train.py --gpu_ids 2 --name grid-cv-${i} --data_dir $data_dir --batchsize 32 --num_epoch 25
done
