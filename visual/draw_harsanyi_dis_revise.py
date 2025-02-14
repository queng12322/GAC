import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20 

sns.set_theme(style="whitegrid", palette="muted", font="Times New Roman", font_scale=1)

data_folder = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/ave_harsanyi_dividend_value_mc/"
output_path = "/mnt/petrelfs/quxiaoye/yuzengqi/OUR_LLM/figure/sorted_harsanyi_plot_with_divider_mc_revise.pdf"

num_layers = 32
layer_files = [f"layer_{i}.log" for i in range(num_layers)]

layer_data = {}

for layer_idx, layer_file in enumerate(layer_files):
    file_path = os.path.join(data_folder, layer_file)
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    combination, harsanyi_value = int(parts[0]), float(parts[1])
                    data.append((combination, harsanyi_value))
        df = pd.DataFrame(data, columns=["Combination", "Harsanyi"])
        layer_data[layer_idx] = df
    else:
        print(f"File {file_path} not found.")

selected_layers = [0, 4, 8, 12, 16, 20, 24, 28]

colors = sns.color_palette("tab10", n_colors=len(selected_layers))

width_inches = 13
height_per_subplot = (width_inches/4) * (5/6)
total_height = height_per_subplot * 2

fig, axes = plt.subplots(2, 4, figsize=(width_inches, total_height))
axes = axes.flatten()

# Adjust spacing between subplots - reduce h_pad and w_pad
plt.tight_layout(rect=[0.03, 0.03, 1, 0.96], h_pad=0.5, w_pad=0.3)  # Reduced from h_pad=1.0, w_pad=0.5

for idx, layer_idx in enumerate(selected_layers):
    if layer_idx in layer_data:
        df = layer_data[layer_idx]
        df_sorted = df.sort_values(by="Harsanyi", ascending=False).reset_index(drop=True)
        
        median_val = df_sorted["Harsanyi"].median()
        df_sorted["Harsanyi"] -= median_val

        df_sorted.rename(columns={"Combination": "Old_Combination"}, inplace=True)
        df_sorted["Combination"] = np.arange(len(df_sorted))  # 重新编号

        ax = axes[idx]

        x_vals = df_sorted["Combination"]
        y_vals = df_sorted["Harsanyi"]

        # Calculate y-axis range and set consistent relative shading
        max_val = y_vals.max()
        min_val = y_vals.min()
        delta = max_val - min_val
        shade_range = 0.1 * delta  # Use 10% of total range for shading

        ax.set_ylim(min_val - 0.05 * delta, max_val + 0.05 * delta)
        
        # Plot main curve
        ax.plot(x_vals, y_vals, color=colors[idx], label=f"Layer {layer_idx}", linewidth=2.5)
        
        # Add horizontal line and shading
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.fill_between(x_vals, -shade_range, shade_range, 
                       color='gray', alpha=0.2)
        
        # Find intersections with y=0
        zero_crossings = np.where(np.diff(np.signbit(y_vals)))[0]
        for cross in zero_crossings:
            # Linear interpolation to find exact x value
            x1, x2 = x_vals[cross], x_vals[cross + 1]
            y1, y2 = y_vals[cross], y_vals[cross + 1]
            x_intersect = x1 + (-y1 * (x2 - x1) / (y2 - y1))
            
            # Plot red x marker at intersection
            ax.plot(x_intersect, 0, 'rx', markersize=8, markeredgewidth=2)

        # Calculate y-axis limits first
        ymin, ymax = ax.get_ylim()

        # Find maximum point
        max_idx = np.argmax(y_vals)  
        max_x = x_vals[max_idx]
        max_y = y_vals[max_idx]

        # Position annotation with straight arrow
        ax.annotate('salient', 
                    xy=(max_x + 0.05 * 2**8, max_y),  # Point slightly right of max value
                    xytext=(max_x + 0.15 * 2**8, max_y - 0.3 * (ymax - ymin)),  # Text position
                    fontsize=20,  # Increased from 12 to 16
                    color='red',
                    arrowprops=dict(
                        facecolor='red',
                        shrink=0.05,
                        width=1,
                        headwidth=8,
                        color='red'
                        # Removed connectionstyle for straight arrow
                    ))

        # Use ymin, ymax for text positioning
        text_y = ymax - 0.05 * (ymax - ymin)

        # Get y-axis limits for positioning
        ymin, ymax = ax.get_ylim()
        text_y = ymax - 0.05 * (ymax - ymin)  # Position closer to top
        
        # Center the text at the top
        ax.text(2**8/2, text_y, f"Layer {layer_idx}", 
                fontsize=20, 
                horizontalalignment='center',  # Center horizontally
                verticalalignment='top')      # Align to top

        ax.set_xticks([0, 2**8])
        ax.set_xticklabels(["0", "$2^n$"], fontsize=18)
        ax.set_yticks([0, max_val])
        ax.set_yticklabels(["0", f"{max_val:.2f}"], fontsize=18)

        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

for i in range(len(selected_layers), len(axes)):
    fig.delaxes(axes[i])

# Add x-axis labels to bottom row
for i in range(4, 8):  # Bottom row indices
    axes[i].set_xlabel("Different interactions", fontsize=20)

# Add y-axis labels to leftmost plots with rotation
axes[0].set_ylabel("w(H)", fontsize=20, rotation=0, labelpad=10)  # Top left subplot
axes[4].set_ylabel("w(H)", fontsize=20, rotation=0, labelpad=10)  # Bottom left subplot

# Adjust spacing to accommodate labels
plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])  # Increased left margin slightly

plt.savefig(output_path, dpi=300, bbox_inches='tight', format='svg')
plt.show()
