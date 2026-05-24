import matplotlib.pyplot as plt
import numpy as np


thresholds = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9]
recall = [68.5, 72.1, 76.5, 78.8, 79.2, 77.0, 65.4, 54.2] 
noise_edges = [407.5 , 88.5, 80, 70, 42.75, 31.25, 18.75, 16]   

fig, ax1 = plt.subplots(figsize=(8, 5))

# 绘制召回率 (主Y轴)
color = 'tab:blue'
ax1.set_xlabel('Semantic Edge Threshold ($\\tau_{sim}$)', fontsize=12) # 使用 LaTeX 语法的 tau
ax1.set_ylabel('Recall@10 (%)', color=color, fontsize=12, fontweight='bold')
ax1.plot(thresholds, recall, marker='o', color=color, linewidth=2.5, markersize=8, label='Recall@10')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# 绘制边数 (次Y轴)
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Average Number of Edges', color=color, fontsize=12, fontweight='bold')  
ax2.plot(thresholds, noise_edges, marker='s', linestyle='--', color=color, linewidth=2.5, markersize=8, label='Edge Count')
ax2.tick_params(axis='y', labelcolor=color)

# 标记最佳点
ax1.axvline(x=0.65, color='gray', linestyle=':', linewidth=2, alpha=0.8)
# 给最佳区间加个轻微的背景高亮
ax1.axvspan(0.6, 0.7, color='gray', alpha=0.1) 
ax1.text(0.655, min(recall) + (max(recall)-min(recall))*0.2, 'Optimal Region', rotation=90, color='dimgray', fontsize=11, fontweight='bold')

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=11)

fig.tight_layout()  


plt.savefig('fig3-4_parameter_sensitivity.pdf', format='pdf', bbox_inches='tight', dpi=300)
print("✅ 图表已保存为 fig3-4_parameter_sensitivity.pdf")