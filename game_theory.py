import os
import time 
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
import re
import string
from collections import Counter
from tqdm import tqdm
from torch import nn
import random
from typing import List, Optional, Tuple, Union
from transformers import LlamaTokenizer,AutoTokenizer,MistralForCausalLM
from transformers import LlamaTokenizer,AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention
from model_aug.llama_modeling_aug import *
from utils.data_utils import *
from utils.data_utils import choices
import pdb
from tools.compute_harsanyi_dividends import *

NUM_LAYER = int(os.environ.get("NUM_LAYER", -1))

tasks = {
    'classification': ['sst2', 'sst5', 'MR', 'SUBJ', 'AGNews', 'TREC', 'CB', 'BoolQ'],
    'multiple_choice': ['hellaswag', 'ARCE', 'PIQA', 'ARCC', 'OB', 'CQA'],
    'question_answer': ['SQuADv1', 'SQuADv2']
}
 
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load(ckpt_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        use_fast=False,
        cache_dir="../",
        padding_side="left",
    )
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage = True,cache_dir="../", torch_dtype=torch.float16, attn_implementation="eager", device_map="auto")
    model.half()
    model.to('cuda')

    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def compute_metrics(args, results: dict, total_num: int) -> float:
    total_acc = 0
    accs = []
    for name, correct in results.items():
        if args.calibrate:
            if args.task_type == "classification":
                if name != "CB":
                    acc = correct / args.num_samples
                else:
                    acc = correct / 250
            elif args.task_type == "multiple_choice":
                if name != "COPA":
                    acc = correct / args.num_samples
                else:
                    acc = correct / 400
            accs.append(acc)
        else:
            acc = correct / total_num
        total_acc += correct
        print("ACC-%s: %.4f" % (name, acc))
    print("ACC-all: %.4f" % (total_acc/total_num))
    try:
        if args.calibrate:
            if args.task_type == "classification":
                with open(args.output_dir, 'a') as file:
                    file.write("ACC-all: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" % \
                        (total_acc/total_num, accs[0], accs[1], accs[2], accs[3], \
                            accs[4], accs[5], accs[6], accs[7], accs[8])) 
            elif args.task_type == "multiple_choice":
                with open(args.output_dir, 'a') as file:
                    file.write("ACC-all: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" % \
                        (total_acc/total_num, accs[0], accs[1], accs[2], accs[3], \
                            accs[4], accs[5], accs[6])) 
    except Exception as e:
        print(f"An error occurred: {e}")         
    return total_acc/total_num
            

def normalize_answer(s):
    """Normalize the answer text."""
    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_tokens = normalize_answer(a_gold).split()
    pred_tokens = normalize_answer(a_pred).split()
    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return int(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

pdb.set_trace = lambda: None

def main(args):
    set_random_seed(42)
    model, tokenizer = load(args.ckpt_dir)
    if args.calibrate:
        print("---------------------Calibration---------------------")
        calibration_dataset = get_calibration_dataset(args.task_type, tasks[args.task_type], args.num_samples)
        final_dataset_prompt = [sample['sentence'] for sample in calibration_dataset]
        final_dataset = calibration_dataset
        correct_counts = {key: 0 for key in tasks[args.task_type]}
    else:
        print("---------------------Eval---------------------")
        print(f"-------------------dataset:{args.dataset}-------------------")
        if args.task_type == "classification":
            dataset = get_classification_dataset(args.dataset)
            eval_dataset = get_formatted_evaluation_classification_dataset(dataset, args.few_shot_number)
        elif args.task_type == "multiple_choice":
            dataset = get_multiple_choice_dataset(args.dataset)
            eval_dataset = get_formatted_evaluation_mc_dataset(dataset, args.few_shot_number)
        final_dataset_prompt = [sample['sentence'] for sample in eval_dataset]
        final_dataset = eval_dataset
        correct_counts = {eval_dataset[0]['name']: 0}

    print("final_dataset_prompt[0]", final_dataset_prompt[0])
    start_time = time.time()
    batch_size = 1
    count = 0
    model.eval()
    all_input_ids = list()
    with torch.no_grad():
        if args.task_type == "classification":
            for idx, batch_input in enumerate(tqdm(batch_split(final_dataset_prompt, batch_size))):
                choices1 = final_dataset[count]['label_choices']
                answer = final_dataset[count]['label']

                encoded_answer = tokenizer(choices1, padding=True, return_tensors='pt')
                encoded_answer = encoded_answer['input_ids'][:, -1].flatten()
                encoded_inputs = prepare_input(tokenizer, batch_input)
                all_input_ids.append(encoded_inputs)
                for head_comb in range(256):
                    path_name = f"/numLayer_{NUM_LAYER}/harsanyi_dividend_headComb_sampleIdx_{idx}.log"
                    game_theory_result_path = args.game_theory_result_path + path_name
                    logits = model(**encoded_inputs, head_comb=head_comb).logits
                    os.makedirs(os.path.dirname(game_theory_result_path), exist_ok=True)
                    logits = logits[0][-1]

                    all_logits = logits[encoded_answer]
                    probabilities = F.softmax(all_logits, dim=-1)

                    log_score = probabilities[answer]
                    if head_comb == 0:
                        log_score = 0
                    with open(game_theory_result_path, 'a') as file:
                        file.write(f"{head_comb} {log_score:.6f}\n")

                count += 1
                compute_harsanyi_dividend_score(args, idx)
        elif args.task_type == "multiple_choice":
            answers = []
            for idx, batch_input in enumerate(tqdm(batch_split(final_dataset_prompt, batch_size))):
                encode_inputs = prepare_input(tokenizer, batch_input)
                label = choices[int(final_dataset[count]['label'])]
                output_label = tokenizer(' ' + label, padding=True, return_tensors='pt')
                output_label = output_label['input_ids'][0][1]
                for head_comb in range(256):
                    path_name = f"/numLayer_{NUM_LAYER}/harsanyi_dividend_headComb_sampleIdx_{idx}.log"
                    game_theory_result_path = args.game_theory_result_path + path_name
                    outputs = model.generate(
                        **encode_inputs, 
                        max_new_tokens = 1, 
                        pad_token_id = tokenizer.pad_token_id, 
                        do_sample=False, 
                        num_beams=1,
                        game_theory_result_path=game_theory_result_path,
                        head_comb=head_comb,
                        output_label=output_label
                        )
                count += 1
                compute_harsanyi_dividend_score(args, idx)
            print("answers", set(answers))
        elif args.task_type == "question_answer":
            num_sample = len(final_dataset)
            for head_comb in range(256):
                count = 0
                total_f1 = 0
                total_exact = 0
                path_name = f"/numLayer_{NUM_LAYER}/harsanyi_dividend_headComb.log"
                game_theory_result_path = args.game_theory_result_path + path_name
                for idx, batch_input in enumerate(tqdm(batch_split(final_dataset_prompt, batch_size))):
                    encode_inputs = prepare_input(tokenizer, batch_input)
                    label = final_dataset[count]['label']
                    input_ids = encode_inputs["input_ids"]
                    input_length = input_ids.shape[1]
                    outputs = model.generate(
                        **encode_inputs, 
                        max_new_tokens=50, 
                        pad_token_id=tokenizer.pad_token_id, 
                        do_sample=False, 
                        num_beams=1,
                        head_comb=head_comb,
                    )
                    generated_tokens = outputs[:, input_length:]
                    pred = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    pred_sentence = pred.split(".")[0] + "."
                    exact = compute_exact(label, pred_sentence)
                    f1 = compute_f1(label, pred_sentence)
                    total_exact += exact
                    total_f1 += f1
                    count += 1
                average_exact = total_exact / num_sample
                average_f1 = total_f1 / num_sample
                # Note, replace this value with the average F1 score under meaningless input.
                if head_comb == 0:
                    average_f1 = 0
                os.makedirs(os.path.dirname(game_theory_result_path), exist_ok=True)
                with open(game_theory_result_path, 'a') as file:
                    file.write(f"{head_comb} {average_f1:.6f}\n")
            compute_harsanyi_dividend_score_question_answer(args, idx)
    end_time = time.time()
    compute_metrics(args, correct_counts, total_num = len(final_dataset))
    print("Total run time: %.2f" % (end_time - start_time))
    return 

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
    # After the game theory computation is complete, execute this code to obtain the heads that need adjustment.

    # If you want to use other types of models, please replace this part.
    if args.do_augmentation:
        print("-------------------Attention Aug-----------------")
        if args.calibrate:
            LlamaAttention.forward = atten_aug_forward_cal_llama
        else:
            LlamaAttention.forward = atten_aug_forward_eval_llama
    else:
        print("-------------------No Attention Aug-----------------")

    main(args)