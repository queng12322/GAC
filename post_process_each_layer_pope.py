import os
import re
import argparse

parser = argparse.ArgumentParser(description="Extract the Accuracy: value and sort it.")
parser.add_argument('--folder_path', type=str, required=True, help="Folder path.")

args = parser.parse_args()

folder_path = args.folder_path

acc_pattern = r"Accuracy:\s*([\d.]+)"

acc_values = {}

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            content = f.read()
            match = re.search(acc_pattern, content)
            if match:
                acc_values[file_name] = float(match.group(1))

sorted_acc_values = sorted(acc_values.items(), key=lambda x: int(x[0].split('_')[-1].split('.')[0]))
acc_values_list = [acc_value for _, acc_value in sorted_acc_values]

print(f"{acc_values_list}")
