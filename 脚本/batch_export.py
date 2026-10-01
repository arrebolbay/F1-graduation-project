#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量下载 2022-2025 摩纳哥站全部车手遥测数据并生成可视化图表。

用法:
    python batch_export.py                    # 全部年份,汇总图+单圈图
    python batch_export.py --skip-per-lap     # 只生成汇总图(快很多)
    python batch_export.py --years 2024 2025  # 只处理指定年份
"""

import argparse
import os
import sys
import time

import fastf1
import pandas as pd

# ---- 路径 ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PROJECT_DIR, "缓存", "f1_cache")
YEARS = [2022, 2023, 2024, 2025]
EVENT = "Monaco"
SESSION_TYPE = "R"


def parse_args():
    p = argparse.ArgumentParser(description="批量导出摩纳哥站遥测数据并绘图")
    p.add_argument("--years", nargs="+", type=int, default=YEARS)
    p.add_argument("--skip-charts", action="store_true",
                   help="只导出 CSV,不生成图表")
    p.add_argument("--skip-per-lap", action="store_true",
                   help="只生成5张汇总图,不生成77张单圈图")
    return p.parse_args()


# ------------------------------------------------------------------ 导出遥测
def export_telemetry(session, driver, year):
    """导出单个车手的遥测 CSV。已存在则直接返回路径,无数据返回 None。"""
    out_dir = os.path.join(PROJECT_DIR, str(year), driver)
    os.makedirs(out_dir, exist_ok=True)
    csv_name = f"{driver.lower()}_monaco_{year}_all_laps_telemetry.csv"
    csv_path = os.path.join(out_dir, csv_name)

    if os.path.exists(csv_path):
        return csv_path

    laps = session.laps.pick_drivers(driver)
    valid = laps.dropna(subset=["LapTime"])
    if valid.empty:
        return None

    parts = []
    for _, lap in valid.iterrows():
        try:
            tel = lap.get_telemetry()
            if tel is not None and not tel.empty:
                tel = tel.copy()
                tel["LapNumber"] = lap["LapNumber"]
                parts.append(tel)
        except Exception:
            pass

    if not parts:
        return None

    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(csv_path, index=False)
    return csv_path


# ------------------------------------------------------------------ 生成图表
def make_charts(csv_path, skip_per_lap):
    """调用 plot_lec_monaco_speed.py 生成图表。"""
    import plot_lec_monaco_speed as plotter

    argv = ["--csv", csv_path]
    if not skip_per_lap:
        argv.append("--per-lap")
    try:
        plotter.main(argv)
    except (SystemExit, Exception) as exc:
        print(f"    图表生成异常: {exc}")


# ------------------------------------------------------------------ 主流程
def main():
    args = parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    sys.path.insert(0, SCRIPT_DIR)

    total_new = 0
    total_skip = 0
    t_all = time.time()

    for year in args.years:
        print(f"\n{'=' * 60}")
        print(f"  {year} 年摩纳哥大奖赛 (正赛)")
        print(f"{'=' * 60}")
        t0 = time.time()

        try:
            session = fastf1.get_session(year, EVENT, SESSION_TYPE)
            session.load(telemetry=True, weather=False, messages=False)
        except Exception as exc:
            print(f"  加载失败,跳过: {exc}")
            continue

        drivers = sorted(session.laps["Driver"].unique())
        print(f"  车手 {len(drivers)} 人 | 总圈数 {session.total_laps}")

        for drv in drivers:
            out_dir = os.path.join(PROJECT_DIR, str(year), drv)
            csv_name = f"{drv.lower()}_monaco_{year}_all_laps_telemetry.csv"
            csv_path_guess = os.path.join(out_dir, csv_name)
            existed = os.path.exists(csv_path_guess)

            csv_path = export_telemetry(session, drv, year)
            if csv_path is None:
                print(f"  [{drv:>3s}] 无有效数据")
                continue

            if existed:
                total_skip += 1
                tag = "已有"
            else:
                total_new += 1
                sz = os.path.getsize(csv_path) // 1024
                tag = f"新导出 {sz}KB"

            if args.skip_charts:
                print(f"  [{drv:>3s}] {tag}")
            else:
                print(f"  [{drv:>3s}] {tag} → 生成图表...")
                make_charts(csv_path, args.skip_per_lap)

        dt = time.time() - t0
        print(f"\n  {year} 完成 ({dt:.0f}s)")

    dt_all = time.time() - t_all
    print(f"\n{'=' * 60}")
    print(f" 全部完成! 总耗时 {dt_all / 60:.1f} 分钟")
    print(f"   新导出 CSV: {total_new}")
    print(f"   已有跳过:   {total_skip}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
