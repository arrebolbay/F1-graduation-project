#!/usr/bin/env python3
r"""
获取 2024 年摩纳哥大奖赛勒克莱尔所有有效圈的遥测数据(需要 fastf1)。
要求: fastf1 >= 3.0, pandas
安装: pip install fastf1 pandas

整理到 F1毕设 之后的路径约定(数据按"年 → 车手"两级存放)
    * 缓存目录: F1毕设\缓存\f1_cache
    * 数据输出: F1毕设\<年份>\<车手>\<车手小写>_monaco_<年份>_all_laps_telemetry.csv
      例如: F1毕设\2024\LEC\lec_monaco_2024_all_laps_telemetry.csv
运行: python f1test.py   (与当前工作目录无关)
"""

import fastf1
import pandas as pd
import os
import sys

# 1. 设置缓存目录并确保其存在(整理到 F1毕设 后统一改用绝对路径)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # ...\F1毕设\脚本
PROJECT_DIR = os.path.dirname(BASE_DIR)                     # ...\F1毕设
CACHE_DIR = os.path.join(PROJECT_DIR, "缓存", "f1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
print(f"缓存已启用，目录: {CACHE_DIR}")

# 2. 设置参数（可根据需要修改）
YEAR = 2024
EVENT = "Monaco"          # 也可用轮次数字，如 7（请以实际赛历为准）
SESSION = "R"             # R = 正赛, Q = 排位, 等
DRIVER = "LEC"            # 勒克莱尔的代码

def main():
    # 3. 加载会话
    try:
        print(f"正在加载 {YEAR}年 {EVENT} {SESSION} 会话...")
        session = fastf1.get_session(YEAR, EVENT, SESSION)
        session.load(telemetry=True, weather=False, messages=False)
        print("会话加载成功")
    except Exception as e:
        print(f"加载失败: {e}")
        print("\n可能原因：2026年摩纳哥大奖赛的数据尚未在fastf1数据源中发布。")
        print("请确认赛事已结束，或尝试更换年份/赛道。")
        sys.exit(1)

    # 4. 筛选勒克莱尔的圈数
    driver_laps = session.laps.pick_driver(DRIVER)
    if driver_laps.empty:
        print(f"错误：未找到车手 {DRIVER} 的数据。")
        sys.exit(1)

    # 只保留有有效圈速的圈（过滤掉进站圈、暖胎圈等）
    valid_laps = driver_laps.dropna(subset=['LapTime'])
    print(f"勒克莱尔共有 {len(valid_laps)} 个有效比赛圈。")

    if len(valid_laps) == 0:
        print("没有有效的圈速数据，退出。")
        sys.exit(1)

    # 5. 遍历所有有效圈，提取遥测数据
    all_telemetry = []
    failed_laps = []

    for idx, lap in valid_laps.iterrows():
        lap_num = lap['LapNumber']
        try:
            telemetry = lap.get_telemetry()
            if telemetry.empty:
                print(f"警告：第 {lap_num} 圈遥测数据为空，跳过")
                continue
            telemetry = telemetry.copy()
            telemetry['LapNumber'] = lap_num   # 增加圈数列，便于区分
            all_telemetry.append(telemetry)
        except Exception as e:
            failed_laps.append(lap_num)
            print(f"获取第 {lap_num} 圈遥测数据失败: {e}")

    if failed_laps:
        print(f"以下圈数获取失败: {failed_laps}")

    if not all_telemetry:
        print("没有获取到任何遥测数据，程序结束。")
        sys.exit(1)

    # 6. 合并所有数据
    combined_df = pd.concat(all_telemetry, ignore_index=True)
    print(f"成功获取 {len(all_telemetry)} 圈的遥测数据，总共 {len(combined_df)} 行记录。")

    # 7. 保存为CSV文件(F1毕设\<年份>\<车手>\)
    out_dir = os.path.join(PROJECT_DIR, str(YEAR), DRIVER)
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir,
                               f"{DRIVER.lower()}_monaco_{YEAR}_all_laps_telemetry.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"数据已保存至: {output_file}")

    # 可选：打印数据摘要
    print("\n数据列: ", combined_df.columns.tolist())
    print("前5行预览:")
    print(combined_df.head())

if __name__ == "__main__":
    main()