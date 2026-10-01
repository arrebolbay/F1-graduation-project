#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
圈速波动 σ 估计(蒙特卡洛噪声标定)
================================
从 FastF1 缓存(2022-2025 摩纳哥正赛)提取每车手每 stint 的有效圈速,
对每个 stint 做线性去趋势(吸收燃油减轻 + 轮胎衰减的确定性部分),
残差即"圈速随机波动"。输出:
    图表/14_圈速波动估计.csv   汇总/分年/分车手/分配方 σ 与正态性检验
结论供 web\engine.py 的 lap_noise_std 默认值使用。

过滤: 绿旗圈(TrackStatus=1)、非进站圈、剔除第 1 圈(发车圈)、有效 LapTime。
"""

import os
import sys

import numpy as np
import pandas as pd
import fastf1
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PROJECT_DIR, "缓存", "f1_cache")
CHART_DIR = os.path.join(PROJECT_DIR, "图表")
YEARS = [2022, 2023, 2024, 2025]
EVENT = "Monaco"

# 2025 摩纳哥 C6/C5/C4 → 保留绝对 ID;2022-2024 SOFT/MEDIUM/HARD → C5/C4/C3
COMPOUND_MAP = {
    "SOFT": "C5", "MEDIUM": "C4", "HARD": "C3",
    "C5": "C5", "C4": "C4", "C3": "C3",
}


def collect_residuals():
    """逐年逐车手提取 stint 内去趋势残差。返回 DataFrame。"""
    rows = []
    for year in YEARS:
        print(f"[{year}] 加载 {EVENT} Race ...")
        session = fastf1.get_session(year, EVENT, "R")
        session.load(laps=True, telemetry=False, weather=False,
                     messages=False)
        laps = session.laps
        for drv, dl in laps.groupby("Driver"):
            df = pd.DataFrame({
                "LapNumber": dl["LapNumber"].values,
                "LapTime_s": dl["LapTime"].dt.total_seconds().values,
                "Compound": dl["Compound"].values,
                "Stint": dl["Stint"].values,
                "PitIn": dl["PitInTime"].notna().values,
                "PitOut": dl["PitOutTime"].notna().values,
                "TrackStatus": dl["TrackStatus"].astype(str).values,
            })
            # 过滤: 有效圈速 / 非进站圈 / 绿旗 / 非第 1 圈
            ok = (
                df["LapTime_s"].notna()
                & ~df["PitIn"] & ~df["PitOut"]
                & df["TrackStatus"].str.startswith("1")
                & (df["LapNumber"] > 1)
            )
            df = df[ok].copy()
            df["Comp"] = df["Compound"].map(COMPOUND_MAP)
            df = df.dropna(subset=["Comp"])
            # 每个 (stint) 线性去趋势
            for stint, g in df.groupby("Stint"):
                g = g.sort_values("LapNumber")
                if len(g) < 5:
                    continue
                x = np.arange(len(g), dtype=float)
                y = g["LapTime_s"].values.astype(float)
                k, b = np.polyfit(x, y, 1)
                resid = y - (k * x + b)
                for r, (_, row) in zip(resid, g.iterrows()):
                    rows.append({
                        "年份": year, "车手": drv, "配方": row["Comp"],
                        "Stint": int(stint), "残差_s": float(r),
                    })
        print(f"[{year}] 完成")
    return pd.DataFrame(rows)


def summarize(df):
    """σ 估计 + 正态性检验(含 1.5IQR 去尾稳健估计)。"""
    out = []
    r = df["残差_s"].values
    q1, q3 = np.percentile(r, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    t = r[(r >= lo) & (r <= hi)]

    def norm_row(vals, label, extra=""):
        sigma = float(np.std(vals, ddof=1))
        stat, p = stats.normaltest(vals)
        skew, kurt = float(stats.skew(vals)), float(stats.kurtosis(vals))
        return {"范围": label, "键": extra, "样本数": len(vals),
                "sigma_s": round(sigma, 4),
                "sigma_10圈累计_s": round(sigma * np.sqrt(10), 3),
                "偏度": round(skew, 3), "峰度": round(kurt, 3),
                "正态检验p值": f"{p:.3g}",
                # normaltest: p > 0.05 → 不能拒绝正态假设
                "结论": "近似正态" if p > 0.05 else "偏离正态(重尾)"}

    out.append(norm_row(r, "汇总(原始)", "ALL"))
    out.append(norm_row(t, "汇总(1.5IQR去尾)", "ALL-TRIM"))
    for year, g in df.groupby("年份"):
        out.append({"范围": "年份", "键": str(year), "样本数": len(g),
                    "sigma_s": round(float(g["残差_s"].std(ddof=1)), 4),
                    "sigma_10圈累计_s": "",
                    "偏度": "", "峰度": "", "正态检验p值": "", "结论": ""})
    for drv, g in df.groupby("车手"):
        out.append({"范围": "车手", "键": drv, "样本数": len(g),
                    "sigma_s": round(float(g["残差_s"].std(ddof=1)), 4),
                    "sigma_10圈累计_s": "",
                    "偏度": "", "峰度": "", "正态检验p值": "", "结论": ""})
    for comp, g in df.groupby("配方"):
        out.append({"范围": "配方", "键": comp, "样本数": len(g),
                    "sigma_s": round(float(g["残差_s"].std(ddof=1)), 4),
                    "sigma_10圈累计_s": "",
                    "偏度": "", "峰度": "", "正态检验p值": "", "结论": ""})
    return pd.DataFrame(out)


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    df = collect_residuals()
    if df.empty:
        print("无有效数据!")
        sys.exit(1)
    summary = summarize(df)
    out_path = os.path.join(CHART_DIR, "14_圈速波动估计.csv")
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    raw_path = os.path.join(CHART_DIR, "14b_圈速残差明细.csv")
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(summary[summary["范围"] == "汇总"].to_string(index=False))
    print(f"\n已保存: {out_path}")
    print(f"已保存: {raw_path}")


if __name__ == "__main__":
    main()