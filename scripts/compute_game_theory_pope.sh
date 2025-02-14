gpu_ids=$CUDA_VISIBLE_DEVICES
IFS=',' read -r -a gpu_ids <<< "$gpu_ids"
echo "gpu_ids: ${gpu_ids[@]}"


layer_ids="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"

layer_ids_array=($layer_ids)

# The number of layers running on each GPU
layers_per_gpu=2

num_gpus=${#gpu_ids[@]}

gpu_index=0

for ((i = 0; i < ${#layer_ids_array[@]}; i++))
do
    layer_id=${layer_ids_array[$i]}

    gpu_id=${gpu_ids[$gpu_index]}

    echo "GPU $gpu_id will run the task for layer_id=$layer_id..."

    export CUDA_VISIBLE_DEVICES=$gpu_id
    export LAYER_NUM=$layer_id
    export NUM_LAYER=$layer_id
    export BETA=0.1
    export THRES=0.1
    export MODIFIED_LAYER="0"
    
    nohup python game_theory_pope.py \
            --model_path '/mnt/petrelfs/quxiaoye/yuzengqi/MODEL/llava-v1.6-mistral-7b-hf' \
            --do_augmentation 1 \
            --data_path /mnt/petrelfs/quxiaoye/yuzengqi/DATA/coco/val2014 \
            --num_samples 500 \
            --conv_model "llava_mistral_instruct" \
            --game_theory_result_path "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_VLM_COCO_adversarial/game_theory_results_pope" \
            --task_type "pope" \
            --pope-type "adversarial" \
            --calibrate 1 > ./output_log/pope/nohup_layer_${layer_id}.log 2>&1 &

    ((gpu_index++))

    if [ $gpu_index -ge $num_gpus ]; then
        gpu_index=0
    fi

    if [ $i -eq 7 ]; then
        echo "Waiting for all tasks to complete at i=$i..."
        wait
    fi   
    if [ $i -eq 15 ]; then
        echo "Waiting for all tasks to complete at i=$i..."
        wait
    fi   
    if [ $i -eq 23 ]; then
        echo "Waiting for all tasks to complete at i=$i..."
        wait
    fi      

    if [ $i -eq 31 ]; then
        echo "Waiting for all tasks to complete at i=$i..."
        wait
    fi       
done

wait
