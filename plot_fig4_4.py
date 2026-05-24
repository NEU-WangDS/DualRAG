import matplotlib.pyplot as plt
import numpy as np


datasets = ['2WikiMultihopQA', 'HotpotQA']

# 1. Direct (闭卷直答，无检索)
direct_em = [26.1, 19.5]

# 2. Native (纯稠密检索)
native_em = [37.2, 35.1]

# 3. IRCoT (迭代检索+思维链)
ircot_em = [42.5, 39.3]

# 4. DualRAG (未加图谱的文本双过程)
dualrag_em = [53.0, 50.6]

# 5. Graph-DualRAG (本文最终完全体)
graph_dualrag_em = [55.8, 53.5]

# ==========================================

x = np.arange(len(datasets))  # 标签的水平位置
width = 0.15  # 调整柱子的宽度，5个柱子总宽为 0.75

fig, ax = plt.subplots(figsize=(10, 6))

# 学术级渐进式配色 (从灰色基线到亮红色最终模型)
colors = ['#A9A9A9', '#4682B4', '#DAA520', '#3CB371', '#B22222']

# 绘制5组柱状图 (依次平移 x 轴坐标)
rects1 = ax.bar(x - 2*width, direct_em, width, label='Direct', color=colors[0], edgecolor='black', linewidth=1.2, zorder=3)
rects2 = ax.bar(x - width, native_em, width, label='Native', color=colors[1], edgecolor='black', linewidth=1.2, zorder=3)
rects3 = ax.bar(x, ircot_em, width, label='IRCoT', color=colors[2], edgecolor='black', linewidth=1.2, zorder=3)
rects4 = ax.bar(x + width, dualrag_em, width, label='DualRAG', color=colors[3], edgecolor='black', linewidth=1.2, zorder=3)
rects5 = ax.bar(x + 2*width, graph_dualrag_em, width, label='Graph-DualRAG (Ours)', color=colors[4], edgecolor='black', linewidth=1.2, zorder=3)

# 添加文本标签和自定义 X 轴
ax.set_ylabel('Exact Match (EM) (%)', fontsize=13, fontweight='bold')
ax.set_title('End-to-End Exact Match across Baselines', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=13, fontweight='bold')

# 设置图例 (分为两行显示，避免拥挤)
ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9, edgecolor='gray')

# 添加底层网格线
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# 动态获取所有数据中的最大值，设定 Y 轴上限，防止柱子和上边缘贴得太紧
all_em_values = direct_em + native_em + ircot_em + dualrag_em + graph_dualrag_em
ax.set_ylim(0, max(all_em_values) * 1.25)

# 在每个柱子上标注具体数值
def autolabel(rects):
    """在每个柱子上方附加一个文本标签，显示其高度（倾斜显示防重叠）。"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 垂直偏移 4 个点
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=0)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

fig.tight_layout()


save_path = 'fig4-4_em_5_baselines.pdf'
plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"✅ 5模型对比图已成功保存为 {save_path}")