#!/usr/bin/env bash
function run_experiment() {
    a=("$@")
    ((last_idx=${#a[@]} - 1))
    copy=${a[last_idx]}
    unset a[last_idx]
    gpu=${a[last_idx-1]}
    unset a[last_idx-1]
    echo "******************"
    echo "copy: $copy"
    echo "gpu: $gpu"
    echo "******************"
    cp run.py run$copy.py
    sed -i "s|INSTANCE =.*|INSTANCE = $copy|g" run$copy.py
    CUDA_VISIBLE_DEVICES=$GPU python -W ignore run$copy.py \
        --dataset 'mp3d' \
        --distance_model_path 'distance_gcn_noise.pt' \
        --goal_model_path 'goal_mlp_noise.pt' \
        --switch_model_path 'switch_mlp_noise.pt' \
        --path_type 'straight' \
        --pose_noise \
        --difficulty 'medium' \
        --actuation_noise  1> ./results/$copy 2>&1
    rm run$copy.py
}
#####
GPU=0
name=0
x1=("0.55")
run_experiment "{x1[@]}" "$GPU" "$name" &
pids[${name}]=$!
#####
GPU=0
name=1
x1=("0.55")
run_experiment "{x1[@]}" "$GPU" "$name" &
pids[${name}]=$!
#####
GPU=0
name=2
x1=("0.55")
run_experiment "{x1[@]}" "$GPU" "$name" &
pids[${name}]=$!
#####
GPU=1
name=3
x1=("0.55")
run_experiment "{x1[@]}" "$GPU" "$name" &
pids[${name}]=$!
#####
GPU=1
name=4
x1=("0.55")
run_experiment "{x1[@]}" "$GPU" "$name" &
pids[${name}]=$!
#####
GPU=1
name=5
x1=("0.55")
run_experiment "{x1[@]}" "$GPU" "$name" &
pids[${name}]=$!
echo "pids are $pids"
for pid in ${pids[*]}; do
    wait $pid
done
