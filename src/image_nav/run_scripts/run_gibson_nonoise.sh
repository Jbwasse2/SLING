#!/usr/bin/env bash
time=$(date +"%Y-%m-%d_%T")
FLAGS='--dataset=gibson --path_type=straight --difficulty=hard --distance_model_path=NRNS_ours.pt --model TopoGCNNRNS --tag NRNS__glue_fix_meera_hardStraight_straightRightOnly --use_glue_rhophi --dont_reuse_poses --switch_threshold 0.0 --number_of_matches 50 --straight_right_only'
FLAGS='--dataset=gibson --path_type=straight --difficulty=hard --distance_model_path=NRNS_ours.pt --model TopoGCNNRNS --tag NRNS__glue_fix_meera_hardStraight_straightRightOnly --use_glue_rhophi --dont_reuse_poses --switch_threshold 0.0 --number_of_matches 50 --straight_right_only'
regex="--tag ([a-zA-Z0-9_]*)"
[[ $FLAGS =~ $regex ]]
tag=${BASH_REMATCH[1]}
save_path="../../results/masterNav/${tag}/${time}/"
mkdir -p $save_path
echo $save_path
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
#    sed -i "s|INSTANCE =.*|INSTANCE = $copy|g" run$copy.py
    sed -i "s|DUMP_JSON =.*|DUMP_JSON = True|g" run$copy.py
    sed -i "s|curr_time =.*|curr_time = '${time}'|g" run$copy.py
    CUDA_VISIBLE_DEVICES=$GPU python -W ignore run$copy.py "$FLAGS" 1> ${save_path}${copy} 2>&1
    rm run$copy.py
}
####
GPU=1
name=4
x1=("0.55")
run_experiment "{x1[@]}" "$GPU" "$name" &
pids[${name}]=$!
#####
#GPU=1
#name=4
#x1=("0.55")
#run_experiment "{x1[@]}" "$GPU" "$name" &
#pids[${name}]=$!
######
#GPU=1
#name=4
#x1=("0.55")
#run_experiment "{x1[@]}" "$GPU" "$name" &
#pids[${name}]=$!
#####
#GPU=1
#name=4
#x1=("0.55")
#run_experiment "{x1[@]}" "$GPU" "$name" &
#pids[${name}]=$!
######
#GPU=1
#name=4
#x1=("0.55")
#run_experiment "{x1[@]}" "$GPU" "$name" &
#pids[${name}]=$!
######
#GPU=1
#name=4
#x1=("0.55")
#run_experiment "{x1[@]}" "$GPU" "$name" &
#pids[${name}]=$!
echo "pids are $pids"
for pid in ${pids[*]}; do
    wait $pid
done
mv ../../results/0 $save_path
mv ../../results/1 $save_path
mv ../../results/2 $save_path
mv ../../results/3 $save_path
mv ../../results/4 $save_path
mv ../../results/5 $save_path
