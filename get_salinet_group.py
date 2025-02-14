import argparse

from game_theory_post_process import post_process

from game_theory_post_process_pope import post_process_pope

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
    parser.add_argument('--output_dir', type=str, default='/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM_plus/calibration_results/cal.log')
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