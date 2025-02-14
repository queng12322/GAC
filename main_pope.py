import argparse
import os
import random
from re import sub
import numpy as np
from sympy import N
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import pdb
from transformers.models.llama.modeling_llama import LlamaAttention
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time 
import json
from PIL import Image
from transformers.models.mistral.modeling_mistral import MistralAttention
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import copy
import sys

from model_aug.llama_modeling_aug import *

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor

# Change the path to your local path.
POPE_PATH = {
    "random": "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_VLM_COCO_adversarial/data/POPE/coco/coco_pope_random.json",
    "popular": "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_VLM_COCO_adversarial/data/POPE/coco/coco_pope_popular.json",
    "adversarial": "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_VLM_COCO_adversarial/data/POPE/coco/coco_pope_adversarial.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model_path", type=str, default="/mnt/petrelfs/quxiaoye/yuzengqi/MODEL/llava-v1.6-mistral-7b-hf", help="")
    # Select the type of prediction data.
    parser.add_argument("--pope-type", type=str, default="popular", help="")
    parser.add_argument("--conv_model", type=str, default="llava_llama_3", help="")
    # The path of the image file
    parser.add_argument("--data_path", type=str, help="data path")
    parser.add_argument('--do_augmentation', type=int, default=0)
    parser.add_argument("--calibrate", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=500)
    # Intermediate result output address for game theory.
    parser.add_argument("--game_theory_result_path", type=str)
    parser.add_argument("--task_type", type=str)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def print_acc(args, pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))

def recorder(line, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    line = line.replace('.', '')
    line = line.replace(',', '')
    words = line.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        pred_list.append(0)
    else:
        pred_list.append(1)
    
    return pred_list


pdb.set_trace = lambda: None

def main():
    pdb.set_trace()
    args = parse_args()
    if args.do_augmentation:
        print("-------------------Attention Aug-----------------")
        if args.calibrate == 1:
            MistralAttention.forward = atten_aug_forward_cal_mistral
        else:
            MistralAttention.forward = atten_aug_forward_eval_mistral
    else:
        print("-------------------No Attention Aug-----------------")
    setup_seeds(42)
    get_aug_indices()

    print('Initializing Model')

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    processor = LlavaNextProcessor.from_pretrained(args.model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage = True, attn_implementation="eager")
    model.eval()
    model.tie_weights()

    args.pope_path = POPE_PATH[args.pope_type]
    questions = [json.loads(q) for q in open(os.path.expanduser(args.pope_path), "r")]

    # Avoid using training data
    questions = questions[args.num_samples:]

    print ("load data finished")
    print("data length:", len(questions))
    print("Start eval...")

    pred_list, pred_list_s, label_list = [], [], []

    YES_LABEL = ["yes", "Yes", "YES", "y", "Y"]
    NO_LABEL = ["no", "No", "NO", "n", "N"]
    
    for line in tqdm(questions):
        image = line["image"]
        image = Image.open(os.path.join(args.data_path, image))
        qs = line["text"]
        label = line["label"]

        if label in YES_LABEL:
            label_list.append(1)
        elif label in NO_LABEL:
            label_list.append(0)
        else:
            print("Unknown label")
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        conv = copy.deepcopy(conv_templates[args.conv_model])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt += "Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        input_token_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            with torch.no_grad():
                pdb.set_trace()
                output = model.generate(**inputs, max_new_tokens=1, pad_token_id=processor.tokenizer.eos_token_id,)
                pdb.set_trace()

        outputs = processor.decode(output[0][input_token_len:], skip_special_tokens=True)

        pred_list = recorder(outputs, pred_list)

    if len(pred_list) != 0:
        print_acc(args, pred_list, label_list)
    if len(pred_list_s) != 0:
        print_acc(args, pred_list_s, label_list)

if __name__ == "__main__":
    main()