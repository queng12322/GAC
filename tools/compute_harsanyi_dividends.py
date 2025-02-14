import os
from itertools import chain, combinations
import math

'''
You need to modify some constants in the code below according to the number of players in your game.
And you also need to modify the following paths to your own.
'''
NUM_LAYER = int(os.environ.get("NUM_LAYER", -1))

def get_subsets(coalition):
    """Generate all subsets of a given coalition."""
    players = [i for i in range(8) if (coalition >> i) & 1]
    subsets = []
    for r in range(len(players) + 1):
        for combo in combinations(players, r):
            subset = sum(1 << i for i in combo)
            subsets.append(subset)
    return subsets

def compute_harsanyi_dividends(v_values):
    """Compute the Harsanyi dividend for all coalitions."""
    w_values = {}
    for coalition in range(1 << 8):
        subsets = get_subsets(coalition)
        coalition_size = bin(coalition).count('1')
        w = 0
        for subset in subsets:
            subset_size = bin(subset).count('1')
            sign = (-1) ** (subset_size - coalition_size)
            v_subset = v_values.get(subset, 0)
            w += sign * v_subset
        w_values[coalition] = w
    return w_values

def compute_harsanyi_dividend_score(args, idx):
    v_values = {}
    v_filepath = args.game_theory_result_path + f"/numLayer_{NUM_LAYER}/harsanyi_dividend_headComb_sampleIdx_{idx}.log"
    if os.path.exists(v_filepath):
        with open(v_filepath, 'r') as f:
            for line in f:
                head_comb, log_odds = line.strip().split(' ')
                v_values[int(head_comb)] = float(log_odds)
    else:
        print(f"The file {v_filepath} does not exist.")
        return

    w_values = compute_harsanyi_dividends(v_values)
    if args.task_type == "classification":
        output_filepath = f"/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_cf/numLayer_{NUM_LAYER}/harsanyi_dividend_score_sampleIdx_{idx}.log"
    else:
        output_filepath = f"/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_mc/numLayer_{NUM_LAYER}/harsanyi_dividend_score_sampleIdx_{idx}.log"
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'a') as file:
        for head_comb, log_score in w_values.items():
            file.write(f"{head_comb} {log_score:.6f}\n")

def compute_harsanyi_dividend_score_question_answer(args, idx):
    v_values = {}
    v_filepath = args.game_theory_result_path + f"/numLayer_{NUM_LAYER}/harsanyi_dividend_headComb.log"
    if os.path.exists(v_filepath):
        with open(v_filepath, 'r') as f:
            for line in f:
                head_comb, log_odds = line.strip().split(' ')
                v_values[int(head_comb)] = float(log_odds)
    else:
        print(f"The file {v_filepath} does not exist.")
        return

    w_values = compute_harsanyi_dividends(v_values)
    output_filepath = f"/mnt/petrelfs/quxiaoye/yuzengqi/GAC/harsanyi_dividend_value_qa/numLayer_{NUM_LAYER}/harsanyi_dividend_score.log"

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'a') as file:
        for head_comb, log_score in w_values.items():
            file.write(f"{head_comb} {log_score:.6f}\n")



def read_value_function(file_path):
    v = [0.0] * 256  # Initialize the value function array with a size of 2^8 = 256
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            comb_num = int(parts[0])
            value = float(parts[1])
            v[comb_num] = value
    return v