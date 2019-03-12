#!/usr/bin/env bash
export PYTHONPATH=/home/cwh/coding/taudl_pyt
export PYTHON=/home/cwh/anaconda3/bin/python
export LD_LIBRAR_PATH=/usr/local/cuda/lib64

source_names=("market" "duke" "grid" "viper" "cuhk")
#source_names=("grid")
target_names=("market" "duke")
#target_names=("market")
source_class_cnts=(751 702 250 630 971)
#source_class_cnts=(250)

for j in "${!source_names[@]}"
do
    for i in "${!target_names[@]}"
    do
        data_dir="/home/cwh/coding/dataset/${target_names[$i]}/pytorch"
        $PYTHON transfer.py --gpu_ids 3 --name ${source_names[$j]} --test_dir $data_dir --batchsize 32 --class_cnt ${source_class_cnts[$j]}
        $PYTHON test.py --gpu_ids 3 --name ${source_names[$j]} --test_dir $data_dir --batchsize 32 --class_cnt ${source_class_cnts[$j]}
        $PYTHON evaluate_gpu.py --name ${source_names[$j]}_${target_names[i]}-test
    done


    for i in {0..9}
    do
        data_dir="/home/cwh/coding/dataset/grid-cv-$i/pytorch"
        $PYTHON transfer.py --gpu_ids 3 --name ${source_names[$j]} --test_dir $data_dir --batchsize 32 --class_cnt ${source_class_cnts[$j]}
        $PYTHON test.py --gpu_ids 3 --name ${source_names[$j]} --test_dir $data_dir --batchsize 32 --class_cnt ${source_class_cnts[$j]}
        $PYTHON evaluate_gpu.py --name ${source_names[$j]}_grid-cv-${i}-test
    done
done