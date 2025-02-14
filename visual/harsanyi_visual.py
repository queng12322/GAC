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
    
    combinations = []
    harsanyi_values = []
    
    for line in lines:
        combination, harsanyi = line.strip().split()
        combinations.append(int(combination))
        harsanyi_values.append(float(harsanyi))
    
    positive_harsanyi_values = [value for value in harsanyi_values if value > 0]
    
    sorted_indices = np.argsort(harsanyi_values)[::-1]
    sorted_harsanyi_values = np.array(harsanyi_values)[sorted_indices]
    
    total_sum = sum(harsanyi_values)

    normalized_harsanyi_values = (sorted_harsanyi_values - min(sorted_harsanyi_values)) / (max(sorted_harsanyi_values) - min(sorted_harsanyi_values))
    
    heatmap_data = normalized_harsanyi_values.reshape(16, 16)
    
    return heatmap_data, total_sum

def plot_heatmap_and_save(heatmap_data, total_sum, ax, i, j, cbar_ax=None):
    heatmap_data_smoothed = gaussian_filter(heatmap_data, sigma=2)
    sns.heatmap(heatmap_data_smoothed, cmap="YlGnBu", annot=False, fmt=".6f", xticklabels=False, yticklabels=False, ax=ax, cbar_ax=cbar_ax)
    ax.set_aspect('equal')
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([cbar.vmin, cbar.vmax])
    cbar.set_ticklabels(["-1.0", "1.0"])
    cbar.set_label('Relative Harsanyi Dividend Score', rotation=270)
    
    ax.annotate(f"Total Sum: {total_sum:.2f}", xy=(0.3, 0.05), xycoords='axes fraction', ha='center', va='center', fontsize=14, color='black')
    if i != -1:
        ax.annotate(f"Layer {i * 4}", xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='center', fontsize=20, color='black')

def create_2x8_heatmap_layout():
    base_path_before_Harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/ave_harsanyi_dividend_value_mc_before"
    base_path_after_Harsanyi = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/ave_harsanyi_dividend_value_mc_after"
    save_base_path = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/figure/harsanyi_heat_map/28"
    
    os.makedirs(save_base_path, exist_ok=True)

    layer_groups = [
        [0, 4, 8, 12, 16, 20, 24, 28],
        [1, 5, 9, 13, 17, 21, 25, 29],
        [2, 6, 10, 14, 18, 22, 26, 30],
        [3, 7, 11, 15, 19, 23, 27, 31]
    ]
    
    for j, layer_list in enumerate(layer_groups):
        fig, axes = plt.subplots(2, 8, figsize=(30, 6.75))
        
        cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])
    
        for i, ax in enumerate(axes[0]):
            layer = layer_list[i]
            file_path = os.path.join(base_path_before_Harsanyi, f"layer_{layer}.log")
            heatmap_data, total_sum = process_layer_file(file_path)
            plot_heatmap_and_save(heatmap_data, total_sum, ax, -1, -1, cbar_ax)
    
        for i, ax in enumerate(axes[1]):
            layer = layer_list[i]
            file_path = os.path.join(base_path_after_Harsanyi, f"layer_{layer}.log")
            heatmap_data, total_sum = process_layer_file(file_path)
            plot_heatmap_and_save(heatmap_data, total_sum, ax, i, j, cbar_ax)
    
        fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1, wspace=0.05, hspace=0.1)
    
        plt.savefig(os.path.join(save_base_path, f"heatmap_2x8_layout_with_sum_{j}.pdf"))
        plt.close()
    
create_2x8_heatmap_layout()
