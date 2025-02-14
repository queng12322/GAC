import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.ndimage import gaussian_filter
import matplotlib.font_manager as fm

fm.fontManager.addfont('/mnt/petrelfs/quxiaoye/yuzengqi/font/times.ttf')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

def process_layer_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    harsanyi_values = [float(line.strip().split()[1]) for line in lines]
    total_sum = sum(harsanyi_values)

    sorted_harsanyi_values = np.array(harsanyi_values)[np.argsort(harsanyi_values)[::-1]]
    normalized_harsanyi_values = (sorted_harsanyi_values - min(sorted_harsanyi_values)) / (max(sorted_harsanyi_values) - min(sorted_harsanyi_values))
    
    heatmap_data = normalized_harsanyi_values.reshape(16, 16)
    return heatmap_data, total_sum

def plot_heatmap_and_save(heatmap_data, total_sum, ax, layer):
    heatmap_data_smoothed = gaussian_filter(heatmap_data, sigma=2)
    ax.set_aspect('equal')

    ax.annotate(f"Layer {layer}", xy=(0.5, -0.1), xycoords='axes fraction', ha='center', fontsize=20, color='black')
    ax.annotate(f"Total Sum: {total_sum:.2f}", xy=(0.25, 0.05), xycoords='axes fraction', ha='center', fontsize=14, color='black')

def create_8x4_heatmap_layout():
    base_path_before_Harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/ave_harsanyi_dividend_value_mc_before"
    base_path_after_Harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/ave_harsanyi_dividend_value_mc_after"
    save_base_path = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/figure/harsanyi_heat_map/28"
    os.makedirs(save_base_path, exist_ok=True)
    
    for batch in range(2):
        fig, axes = plt.subplots(8, 4, figsize=(16, 30))
        
        layer_start = batch * 16
        for j in range(4):
            for i in range(4):
                layer = layer_start + j * 4 + i
                
                file_path_before = os.path.join(base_path_before_Harsanyi, f"layer_{layer}.log")
                file_path_after = os.path.join(base_path_after_Harsanyi, f"layer_{layer}.log")
                
                heatmap_before, total_before = process_layer_file(file_path_before)
                heatmap_after, total_after = process_layer_file(file_path_after)
                
                plot_heatmap_and_save(heatmap_before, total_before, axes[2 * i, j], layer)
                plot_heatmap_and_save(heatmap_after, total_after, axes[2 * i + 1, j], layer)
        
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.15)
        plt.savefig(os.path.join(save_base_path, f"heatmap_8x4_layout_{layer_start}_{layer_start + 15}.pdf"))
        plt.close()

create_8x4_heatmap_layout()
