r"""快速查看 F1 毕设遥测 CSV 的数据概况。

数据按"年 → 车手"存放于 F1毕设\<年份>\<车手>\ 下。
"""

import glob
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # ...\F1毕设\脚本
PROJECT_DIR = os.path.dirname(BASE_DIR)                        # ...\F1毕设
CSV_NAME = "lec_monaco_2024_all_laps_telemetry.csv"
DEFAULT_CSV = os.path.join(PROJECT_DIR, "2024", "LEC", CSV_NAME)


def find_csv():
    r"""按优先级查找遥测 CSV(F1毕设\<年份>\<车手>\)。"""
    patterns = [DEFAULT_CSV,
                os.path.join(PROJECT_DIR, "*", "*", CSV_NAME),
                os.path.join(PROJECT_DIR, "*", "*", "*monaco*telemetry*.csv"),
                os.path.join(BASE_DIR, CSV_NAME),
                os.path.join(os.getcwd(), CSV_NAME)]
    for pattern in patterns:
        if os.path.isfile(pattern):
            return pattern
        matches = sorted(p for p in glob.glob(pattern) if os.path.isfile(p))
        if matches:
            return matches[0]
    raise SystemExit(r"找不到遥测 CSV,请放到 F1毕设\<年份>\<车手>\ 下。")


# 允许命令行第一个参数指定 CSV 路径
if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    CSV_PATH = sys.argv[1]
else:
    CSV_PATH = find_csv()

print(f"读取文件: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print(df.head())
print(df.info())

