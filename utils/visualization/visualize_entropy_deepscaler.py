import json
import tqdm
import random
from collections import OrderedDict
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import os

# 假设log文件存储在logs目录下，每个文件后缀是.log

df = pd.read_csv('utils/visualization/logs/entropy.csv')  # 替换为你的文件路径

plt.figure(figsize=(10, 6))

col_idxes = [1, 4]

max_len = 800
for col_idx in col_idxes:
    if col_idx == 1:
        name = "GMPO"
        color = (0, 0.8, 0.2)
    elif col_idx == 4:
        name = "GRPO"
        color = (1, 0, 0)
    entropy = df[df.columns[col_idx]].tolist()
    entropy = entropy[:800]
    steps = list(range(len(entropy)))

    # 绘制平滑后的 logprobs_diff_max 曲线
    plt.plot(steps, entropy, label=f"{name}", color=color, linewidth=3)


# 添加图例
plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
# plt.xlabel('Steps')
# plt.ylabel('logprobs_diff')
# plt.title('Logprobs Diff Max and Min')

# 保存图像
plt.savefig('logs/entropy_deepscaler.png', dpi=600, bbox_inches='tight')
plt.show()




