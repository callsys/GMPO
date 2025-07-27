import json
import tqdm
import random
from collections import OrderedDict
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import os

# 假设log文件存储在logs目录下，每个文件后缀是.log
log_dir = 'utils/visualization/logs'  # log文件的目录
log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.log')]

# 创建一个空的DataFrame
data = OrderedDict()

# 解析每个log文件
for log_file in tqdm.tqdm(log_files):
    with open(log_file, 'r') as f:
        log_content = f.read()
        log_content = log_content.replace('}{', '},{')
        log_content = log_content.replace('\n', '') 
        log_content = f'[{log_content}]'
        
        try:
            log_data = json.loads(log_content)
            data[log_file] = log_data[1:]
        except json.JSONDecodeError:
            print(f"Error parsing file {log_file}")

def random_color():
    return (random.random(), random.random(), random.random())


plt.figure(figsize=(10, 6))

def func(key):
    key = key[0]
    res = 0
    if "gmpo" in key:
        res += 10
    
    if "seqclip" in key:
        res -= 1

    if "clip_wider" in key:
        res -= 1
    
    return res

data = OrderedDict(sorted(data.items(), key=func))

max_len = 400
for name, value in data.items():
    if "gmpo_7B.log" in name:
        name = "GMPO"
        color = (0, 0.8, 0.2)
    elif "grpo_7B.log" in name:
        name = "GRPO"
        color = (1, 0, 0)
    else:
        continue
    reward = [el["actor/rewards"] if el["actor/rewards"]<300 else 300 for el in value[:max_len] if "actor/rewards" in el]
    # response_length = [el["eval/average/response_tok_len"] for el in value[:max_len] if "eval/average/accuracy" in el]
    # steps = [el["train/learning_round"] for el in value[:max_len] if "eval/average/accuracy" in el]

    steps = list(range(len(reward)))
    reward = savgol_filter(reward, window_length=11, polyorder=3)
    # 绘制平滑后的 logprobs_diff_max 曲线
    plt.plot(steps, reward, label=f"{name}", color=color, linewidth=3)


# 添加图例
plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
# plt.xlabel('Steps')
# plt.ylabel('logprobs_diff')
# plt.title('Logprobs Diff Max and Min')

# 保存图像
plt.savefig('logs/reward_math.png', dpi=600, bbox_inches='tight')
plt.show()




