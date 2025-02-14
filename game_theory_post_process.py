import os
import numpy as np
import torch

import pdb

def post_process(args, data_len):
    # Set the file path.
    if args.task_type == "classification":
        file_path_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/harsanyi_dividend_value_cf"
        file_path_shapley = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/shapley_value_cf"
        output_folder_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/harsanyi_dividend_value_cf/ave_harsanyi_dividend_value"
        output_folder_shapley = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/shapley_value_cf/ave_shapley_value"
        result_file_path = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/result_cf"
    elif args.task_type == "multiple_choice":
        file_path_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/harsanyi_dividend_value_mc"
        file_path_shapley = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/shapley_value_mc"
        output_folder_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/harsanyi_dividend_value_mc/ave_harsanyi_dividend_value"
        output_folder_shapley = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/shapley_value_mc/ave_shapley_value"
        result_file_path = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/result_mc"
    elif args.task_type == "question_answer":
        file_path_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/harsanyi_dividend_value_qa"
        file_path_shapley = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/shapley_value_qa"
        output_folder_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/harsanyi_dividend_value_qa/ave_harsanyi_dividend_value"
        output_folder_shapley = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/shapley_value_qa/ave_shapley_value"
        result_file_path = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/result_qa"
    os.makedirs(output_folder_harsanyi, exist_ok=True)
    os.makedirs(output_folder_shapley, exist_ok=True)
    os.makedirs(result_file_path, exist_ok=True)

    max_values = []

    for layer in range(32):
        layer_folder = os.path.join(file_path_harsanyi, f'numLayer_{layer}')

        combo_results = {i: [] for i in range(256)}

        for sample_id in range(data_len):
            if args.task_type == "question_answer":
                sample_path = os.path.join(layer_folder, f'harsanyi_dividend_score.log')  
            else:
                sample_path = os.path.join(layer_folder, f'harsanyi_dividend_score_sampleIdx_{sample_id}.log')  

            with open(sample_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    combo_id, score = line.strip().split()
                    combo_id = int(combo_id)
                    score = float(score)

                    combo_results[combo_id].append(score)
            
            if args.task_type == "question_answer":
                break
        pdb.set_trace()
        avg_scores = {combo_id: np.mean(scores) for combo_id, scores in combo_results.items()}

        max_combo_id = max(avg_scores, key=avg_scores.get)


        head_comb = max_combo_id  # The best head number is the maximum combination number.
        mask_values = torch.tensor([1 if (head_comb >> i) & 1 else 0 for i in range(8)])
        # Find the heads that need adjustment.
        adjustable_heads = [i for i, val in enumerate(mask_values) if val == 1]

        print(f'Layer {layer} - Heads to Adjust: {adjustable_heads}')
        max_value = avg_scores[max_combo_id]

        max_values.append(f'{max_combo_id} {max_value}')

        output_file_path = os.path.join(output_folder_harsanyi, f'layer_{layer}.log')

        with open(output_file_path, 'w') as output_file:
            for combo_id, avg_score in avg_scores.items():
                output_file.write(f'{combo_id} {avg_score}\n')

        print(f'Results for Layer {layer} saved to {output_file_path}')

    harsanyi_result_file_path = os.path.join(result_file_path, 'harsanyi_result.log')

    with open(harsanyi_result_file_path, 'w') as result_file:
        result_file.write("\n".join(max_values))

    print(f"Max values for all layers saved to {result_file_path}")


    for layer in range(32):
        layer_folder = os.path.join(file_path_shapley, f'numLayer_{layer}')

        # Initialize a dictionary to store all Shapley values for each player.
        shapley_results = {i: [] for i in range(8)}

        for sample_id in range(data_len):
            if args.task_type == "question_answer":
                sample_path = os.path.join(layer_folder, f'shapley_score.log')
            else:
                sample_path = os.path.join(layer_folder, f'shapley_score_sampleIdx_{sample_id}.log')

            if not os.path.exists(sample_path):
                print(f"File {sample_path} does not exist.")
                continue

            with open(sample_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    player_id, value = parts
                    player_id = int(player_id)
                    value = float(value)

                    shapley_results[player_id].append(value)

            if args.task_type == "question_answer":
                break
        # Compute the average Shapley value for each player.
        avg_shapley_values = {player_id: np.mean(values) if values else 0.0 for player_id, values in shapley_results.items()}

        output_file_path = os.path.join(output_folder_shapley, f'layer_{layer}.log')

        with open(output_file_path, 'w') as output_file:
            for player_id, avg_value in avg_shapley_values.items():
                output_file.write(f'{player_id} {avg_value}\n')

        print(f'Average Shapley values for Layer {layer} saved to {output_file_path}')

