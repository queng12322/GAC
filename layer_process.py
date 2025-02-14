import argparse

parser = argparse.ArgumentParser(description="Process the layer information from user input.")
parser.add_argument('--best_layer', type=str, help="Input data (e.g., 24 or 0 24).")

args = parser.parse_args()

modified_layer = [str(i) for i in range(32)]

input_data = args.best_layer.strip()

input_list = input_data.split()

modified_layer = [s for s in modified_layer if s not in input_list]

result = [input_data.strip() + " " + s for s in modified_layer]

formatted_result = ','.join(f'{item.strip()}' for item in result)

print(formatted_result)