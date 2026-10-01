r"""2024 摩纳哥站 维斯塔潘 vs 勒克莱尔 正赛圈速对比(需要 fastf1)。

整理到 F1毕设 之后
    * 缓存目录: F1毕设\缓存\f1_cache
    * 输出图片: F1毕设\2024\VER与LEC圈速对比.png
运行: python F1test2.py   (与当前工作目录无关)
"""

import os

import fastf1
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ---- 路径(基于脚本位置,与当前工作目录无关) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # ...\F1毕设\脚本
PROJECT_DIR = os.path.dirname(BASE_DIR)                     # ...\F1毕设
CACHE_DIR = os.path.join(PROJECT_DIR, "缓存", "f1_cache")
CHART_DIR = os.path.join(PROJECT_DIR, "2024")               # 多年份/多车手对比图放 F1毕设\<年份>\
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# 中文字体,避免图上的中文变成方框
for _name in ("Microsoft YaHei", "SimHei", "SimSun"):
    try:
        fm.findfont(_name, fallback_to_default=False)
    except Exception:
        continue
    plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
    break
plt.rcParams["axes.unicode_minus"] = False

# 启用缓存(fastf1 要求传入字符串路径)
fastf1.Cache.enable_cache(CACHE_DIR)
print(f"缓存目录: {CACHE_DIR}")

# 加载2024年摩纳哥站正赛
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()

# 列名 'LapTime' 存储的是 timedelta 对象
laps = session.laps[['Driver', 'LapNumber', 'LapTime', 'Compound']]

# 删除空值
laps = laps.dropna(subset=['LapTime'])

# 把 LapTime 转换成秒数（浮点数），便于画图
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

# 选两位车手
drivers_to_plot = ['VER', 'LEC']

plt.figure(figsize=(12, 6))
for driver in drivers_to_plot:
    driver_laps = laps[laps['Driver'] == driver]
    plt.scatter(driver_laps['LapNumber'], driver_laps['LapTimeSeconds'],
                label=driver, s=30, alpha=0.7)

plt.xlabel('圈数 (Lap Number)')
plt.ylabel('圈速 (s)')
plt.title('2024 摩纳哥站 - 关键车手圈速衰减对比')
plt.legend()
plt.grid(True)

out_file = os.path.join(CHART_DIR, "06_VER与LEC圈速对比.png")
plt.savefig(out_file, dpi=150, bbox_inches="tight")
print(f"图片已保存至: {out_file}")
plt.show()

# 查看数据前5行
print(laps[['Driver', 'LapNumber', 'LapTimeSeconds', 'Compound']].head())
