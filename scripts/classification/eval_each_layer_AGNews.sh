dataset_names="AGNews"
augmentations="1"
gpu_ids=$CUDA_VISIBLE_DEVICES

IFS=',' read -r -a gpu_ids <<< "$gpu_ids"
echo "gpu_ids: ${gpu_ids[@]}"
best_layer=""

folder_path=""
baseline=73.16



for repeat in {1..32}
do
    gpu_index=0
    task_count=0
    rubbish_folder="./rubbish${repeat}"
    echo "===== Starting Experiment $repeat ====="
    echo "best_layer: $best_layer"
    modified_layer=$(python layer_process.py --best_layer "$best_layer")
    echo "modified_layer: $modified_layer"
    IFS=',' read -r -a modified_layer_array <<< "$modified_layer"

    for layer in "${modified_layer_array[@]}"
    do
        for dataset_name in $dataset_names
        do
            mkdir -p ./cf_eval_output_log_${dataset_name}
            folder_path="./cf_eval_output_log_${dataset_name}"
            for augmentation in $augmentations
            do
                current_gpu=${gpu_ids[$gpu_index]}
                layer_with_underscore=${layer// /_}
                echo "-----Testing dataset: $dataset_name; If augmentation: $augmentation on GPU $current_gpu-----"
                export CUDA_VISIBLE_DEVICES=$current_gpu
                export BETA=0.1
                export THRES=0.1
                export EPSILON=1e-10
                export MODIFIED_LAYER="$layer"

                nohup python main.py \
                    --ckpt_dir "/mnt/petrelfs/quxiaoye/yuzengqi/MODEL/Llama-3.1-8B-Instruct" \
                    --calibrate 0 \
                    --do_augmentation $augmentation \
                    --task_type 'classification' \
                    --dataset $dataset_name > ./cf_eval_output_log_${dataset_name}/nohup_${layer_with_underscore}_aug_${augmentation}.log 2>&1 &

                task_count=$((task_count + 1))

                if (( task_count % 4 == 0 )); then
                    gpu_index=$(((gpu_index + 1)))
                fi

                if ((gpu_index == 8)); then
                    wait
                    gpu_index=0
                fi
                gpu_index=$(($gpu_index % ${#gpu_ids[@]}))
            done
        done
    done
    wait

    modified_results=$(python post_process_each_layer.py --folder_path "$folder_path")
    max_index=-1
    echo "modified_results: $modified_results"
    processed_results=$(echo "$modified_results" | tr -d '[],' | tr ' ' '\n' | awk -v baseline=$baseline '
    BEGIN { max=-999; max_index=-1; }
    {
        processed_value = $1 * 100 - baseline;
        if (processed_value > max) {
            max = processed_value;
            max_index = NR-1;
        }
        print processed_value;
    }
    END {
        print "Maximum value: " max;
        print max_index;
    }')
    max_index=$(echo "$processed_results" | tail -n 1)
    echo "Accuracy improvement: ${processed_results[@]}"

    best_layer="${modified_layer_array[$max_index]}"
    echo "best_layer: $best_layer"

    if [ ! -d "$rubbish_folder" ]; then
        mkdir -p "$rubbish_folder"
        echo "Create the directory $rubbish_folder."
    fi
    if [ -d "$folder_path" ]; then
        echo "Moving $folder_path to $rubbish_folder."
        mv "$folder_path" "$rubbish_folder"
        echo "$folder_path has been successfully moved to $rubbish_folder."
    else
        echo "Folder does not exist: $folder_path."
    fi
done
