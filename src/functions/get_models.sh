#!/usr/bin/env bash
mkdir -p ../../results/nn_symm/
mkdir -p ../../results/nn_id/
mkdir -p ../../results/nn_tri/
function run_experiment() {
    a=("$@")
    ((last_idx=${#a[@]} - 1))
    copy=${a[last_idx]}
    unset a[last_idx]
    gpu=${a[last_idx-1]}
    unset a[last_idx-1]
    weight="${a[@]}"
    echo "******************"
    echo "copy: $copy"
    echo "gpu: $gpu"
    echo "weight: $weight"
    echo "******************"
    cp ./feat_pred_fuc/run.py ./feat_pred_fuc/run$copy.py
    sleep 5
#    sed -i "s|'clustered_graph|'clustered_graph$copy|" ./feat_pred_fuc/run$copy.py
#    CUDA_VISIBLE_DEVICES=$GPU python feat_pred_fuc/run$copy.py --dataset gibson --intra_identity --id_weight $weight --batch_size 32 --epoch 30 --distance_out --intra_loss --run_name ng_id --tag $weight --early_stopping 1000 1> ../../results/nn_id/${weight}.txt 2>&1
    CUDA_VISIBLE_DEVICES=$GPU python feat_pred_fuc/run$copy.py --dataset gibson --intra_triangle --tri_weight $weight --batch_size 32 --epoch 30 --distance_out --intra_loss --run_name ng_tri --tag $weight --early_stopping 1000 1> ../../results/nn_tri/${weight}.txt 2>&1
#    CUDA_VISIBLE_DEVICES=$GPU python feat_pred_fuc/run$copy.py --dataset gibson --intra_symm --symm_weight $weight --batch_size 32 --epoch 30 --distance_out --intra_loss --run_name ng_symm --tag $weight --early_stopping 1000 1> ../../results/nn_symm/${weight}.txt 2>&1
    rm ./feat_pred_fun/run$copy.py
}
####
GPU=0
name=21
x1=("1.0")
run_experiment "${x1[@]}" "$GPU" "$name" &
pids[${name}]=$!
sleep 900
####
#GPU=0
#name=22
#x2=("1000.0")
#run_experiment "${x2[@]}" "$GPU" "$name" &
#pids[${name}]=$!
#sleep 900
####
GPU=0
name=23
x4=("2.0")
run_experiment "${x4[@]}" "$GPU" "$name" &
pids[${name}]=$!
sleep 900
GPU=0
name=24
x6=("8.0")
run_experiment "${x6[@]}" "$GPU" "$name" &
pids[${name}]=$!
sleep 900
GPU=0
name=25
x8=("32.0")
run_experiment "${x8[@]}" "$GPU" "$name" &
pids[${name}]=$!
sleep 900
#####
GPU=0
name=26
x9=("100.0")
run_experiment "${x9[@]}" "$GPU" "$name" &
pids[${name}]=$!
sleep 900
