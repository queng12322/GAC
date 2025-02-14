# Define the total number of layers and the number of layers per batch
start_layer=0
end_layer=31
batch_size_per_gpu=4

gpu_ids=$CUDA_VISIBLE_DEVICES
IFS=',' read -r -a gpu_ids <<< "$gpu_ids"
echo "gpu_ids: ${gpu_ids[@]}"

num_gpus=${#gpu_ids[@]}
total_layers=$((end_layer - start_layer + 1))
total_batches=$(( (total_layers + batch_size_per_gpu * num_gpus - 1) / (batch_size_per_gpu * num_gpus) ))

for batch_id in $(seq 0 $((total_batches - 1))); do
    echo "Start running batch task $batch_id"

    for index in "${!gpu_ids[@]}"; do

        gpu_id=${gpu_ids[$index]}

        start_index=$((batch_id * num_gpus * batch_size_per_gpu + index * batch_size_per_gpu))
        start_layer_id=$((start_layer + start_index))
        end_layer_id=$((start_layer_id + batch_size_per_gpu - 1))


        if [ $start_layer_id -gt $end_layer ]; then
            continue
        fi
        if [ $end_layer_id -gt $end_layer ]; then
            end_layer_id=$end_layer
        fi

        echo "GPU $gpu_id will run the task of layer_id from $start_layer_id to $end_layer_id"

        for layer_id in $(seq $start_layer_id $end_layer_id); do
            echo "Calculating the task of layer_id=$layer_id (using GPU $gpu_id)"

            export CUDA_VISIBLE_DEVICES=$gpu_id
            export LAYER_NUM=$layer_id
            export MASK_METHOD=1
            export EPSILON=1e-10
            export NUM_LAYER=$layer_id
            export MODIFIED_LAYER="0"

            nohup python game_theory.py \
                   --ckpt_dir "/mnt/petrelfs/quxiaoye/yuzengqi/MODEL/Llama-3.1-8B-Instruct" \
                   --calibrate 1 \
                   --do_augmentation 1 \
                   --num_samples 900 \
                   --output_dir "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/calibration_results/cal_layer.log" \
                   --game_theory_result_path "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/game_theory_results_cf/" \
                   --task_type 'classification' > ./output_log/cf/nohup_layer_${layer_id}.log 2>&1 &
        done
    done

    wait
    echo "Batch Task $batch_id Completed"
done


echo "All tasks have been completed!"
