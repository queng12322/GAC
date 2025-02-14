import os
import numpy as np
import torch

import argparse

def post_process(args, data_len):
    # Set the file path.
    if args.task_type == "classification":
        file_path_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_cf"
        output_folder_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_cf/ave_harsanyi_dividend_value"
        output_folder_shapley = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/shapley_value_cf/ave_shapley_value"
        result_file_path = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/result_cf"
    elif args.task_type == "multiple_choice":
        file_path_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_mc"
        output_folder_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_mc/ave_harsanyi_dividend_value"
        output_folder_shapley = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/shapley_value_mc/ave_shapley_value"
        result_file_path = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/result_mc"
    elif args.task_type == "question_answer":
        file_path_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_qa"
        output_folder_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_qa/ave_harsanyi_dividend_value"
        output_folder_shapley = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/shapley_value_qa/ave_shapley_value"
        result_file_path = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/result_qa"
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

    print(f"Max Harsanyi Dividend values for all layers saved to {result_file_path}")




def post_process_pope(args, data_len):
    if args.task_type == "pope":
        file_path_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_pope"
        output_folder_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_pope/ave_harsanyi_dividend_value"
        output_folder_shapley = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/shapley_value_pope/ave_shapley_value"
        result_file_path = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/result_pope"
    elif args.task_type == "mme":
        file_path_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_mme"
        output_folder_harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_mme/ave_harsanyi_dividend_value"
        output_folder_shapley = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/shapley_value_mme/ave_shapley_value"
        result_file_path = "/mnt/petrelfs/quxiaoye/yuzengqi/GAC/result_mme"
    os.makedirs(output_folder_harsanyi, exist_ok=True)
    os.makedirs(output_folder_shapley, exist_ok=True)
    os.makedirs(result_file_path, exist_ok=True)
    max_values = []

    for layer in range(32):
        layer_folder = os.path.join(file_path_harsanyi, f'numLayer_{layer}')

        combo_results = {i: [] for i in range(256)}

        for sample_id in range(data_len):
            sample_path = os.path.join(layer_folder, f'harsanyi_dividend_score_sampleIdx_{sample_id}.log')  

            with open(sample_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    combo_id, score = line.strip().split()
                    combo_id = int(combo_id)
                    score = float(score)

                    combo_results[combo_id].append(score)

        avg_scores = {combo_id: np.mean(scores) for combo_id, scores in combo_results.items()}
        max_combo_id = max(avg_scores, key=avg_scores.get)

        head_comb = max_combo_id
        mask_values = torch.tensor([1 if (head_comb >> i) & 1 else 0 for i in range(8)])
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

    print(f"Max Harsanyi Dividend values for all layers saved to {result_file_path}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # Model address
    parser.add_argument('--ckpt_dir', type=str, required=True, default="/mnt/petrelfs/quxiaoye/yuzengqi/MODEL/Llama-3.1-8B-Instruct")
    # Whether to compute the Harsanyi dividend.
    parser.add_argument('--calibrate', type=int, default=1)
    # The dataset to be evaluated.
    parser.add_argument('--dataset', type=str, default='sst2')
    # Whether to use GAC.
    parser.add_argument('--do_augmentation', type=int, default=0)
    # Number of game theory samples from each dataset
    parser.add_argument('--num_samples', type=int, default=900)
    # Result output file.
    parser.add_argument('--output_dir', type=str)
    # Type of task to be evaluated.
    parser.add_argument('--task_type', type=str, default='multiple_choice')
    # Intermediate result output address for game theory.
    parser.add_argument('--game_theory_result_path', type=str)

    args = parser.parse_args()
    print("args", args)
    # Please replace with the actual number of samples used
    if args.task_type == "classification":
        post_process(args, 6550)
    elif args.task_type == "multiple_choice":
        post_process(args, 5400)
    elif args.task_type == "question_answer":
        post_process(args, 200)
    elif args.task_type == "pope":
        post_process_pope(args, 500)