#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史回溯验证: 2024 摩纳哥站红旗案例
===================================
2024 摩纳哥正赛第 1 圈发生严重事故,红旗中断后重新发车。红旗期间
可在车房内免费换胎 —— 模型的红旗场景应给出"免费换胎"主导建议。

本脚本用真实比赛数据核验:
  1. 识别红旗圈与重启圈(TrackStatus);
  2. 统计各车手在红旗重启处的换胎行为(stint 边界)与全场 stint 结构;
  3. 对照模型红旗场景(Undercut 框架的 red 分支)建议与 DP 最优策略;
  4. 输出对比表与策略时间线图。

产出: 图表/18_历史回溯_2024红旗案例.csv / 18_历史回溯_2024红旗案例.png
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CHART_DIR = os.path.join(PROJECT_DIR, "图表")
CACHE_DIR = os.path.join(PROJECT_DIR, "缓存", "f1_cache")
sys.path.insert(0, os.path.join(PROJECT_DIR, "web"))
import engine  # noqa: E402

warnings.filterwarnings("ignore")
for _n in ("Microsoft YaHei", "SimHei", "SimSun"):
    try:
        fm.findfont(_n, fallback_to_default=False)
    except Exception:
        continue
    plt.rcParams["font.sans-serif"] = [_n, "DejaVu Sans"]
    break
plt.rcParams["axes.unicode_minus"] = False

COMP_COLORS = {"SOFT": "#e10600", "MEDIUM": "#ffd24d", "HARD": "#b8bdc7",
               "C5": "#e10600", "C4": "#ffd24d", "C3": "#b8bdc7"}


def load_2024():
    import fastf1
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    sess = fastf1.get_session(2024, "Monaco", "R")
    sess.load(telemetry=False, weather=False, messages=False)
    laps = sess.laps.copy()
    laps["LapTimeS"] = laps["LapTime"].dt.total_seconds()
    return laps


def find_red_laps(laps):
    ts = laps.groupby("LapNumber")["TrackStatus"].agg(
        lambda s: "|".join(sorted(set(str(x) for x in s.dropna()))))
    red = [int(n) for n, v in ts.items() if "5" in v]
    sc = [int(n) for n, v in ts.items() if "4" in v and "5" not in v]
    return sorted(red), sorted(sc)


def stint_table(laps):
    rows = []
    for drv, g in laps.groupby("Driver"):
        g = g.sort_values("LapNumber")
        for stint, gg in g.groupby("Stint"):
            rows.append({
                "Driver": drv,
                "Stint": int(stint),
                "Compound": str(gg["Compound"].iloc[0]),
                "StartLap": int(gg["LapNumber"].min()),
                "EndLap": int(gg["LapNumber"].max()),
                "Laps": len(gg),
                "Fresh": bool(gg["FreshTyre"].iloc[0]) if "FreshTyre" in gg else None,
            })
    return pd.DataFrame(rows)


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    print("=" * 55)
    print(" 历史回溯验证: 2024 摩纳哥红旗案例")
    print("=" * 55)
    laps = load_2024()
    red_laps, sc_laps = find_red_laps(laps)
    restart_lap = (max(red_laps) + 1) if red_laps else None
    print(f"红旗圈: {red_laps} | 纯SC圈: {sc_laps[:10]} | 重启圈: {restart_lap}")

    stints = stint_table(laps)
    drivers = sorted(stints["Driver"].unique())

    # 各车手在红旗重启处是否换胎(stint 边界恰好落在红旗/重启圈)
    rows = []
    changed = 0
    for drv in drivers:
        sd = stints[stints["Driver"] == drv].sort_values("Stint")
        boundaries = sd["StartLap"].tolist()[1:]        # 每段起始圈(首段除外)
        red_change = any(b <= (restart_lap or 0) + 1 for b in boundaries)
        n_stint = len(sd)
        if red_change:
            changed += 1
        rows.append({"车手": drv,
                     "红旗重启处换胎": "是" if red_change else "否",
                     "全场stint数": n_stint,
                     "stint结构": " | ".join(
                         f"{r.Compound}({r.StartLap}-{r.EndLap})"
                         for r in sd.itertuples())})
    total_drivers = len(drivers)
    print(f"红旗重启处换胎车手: {changed}/{total_drivers}")

    # ---- 模型对照: 红旗场景建议
    model = engine.run_simulation(
        scenario="red", my_driver="LEC", my_compound="C5", my_age=2,
        rival_driver="VER", rival_compound="C5", rival_age=2,
        current_lap=3, gap_s=0.0, n_sim=5000, auto_fit=True, seed=7)
    rec = model["recommendation"]
    branch_txt = " | ".join(
        f"{b['label']}={b['success_rate']}%" for b in model["branches"])
    print(f"模型红旗场景: {branch_txt}")
    print(f"模型建议: {rec['text'][:80]}")
    agree = "是" if (changed / max(total_drivers, 1)) >= 0.5 and \
        rec["branch"] == "change" else "否"

    rows.append({"车手": "—模型—", "红旗重启处换胎": f"建议:{rec['label']}",
                 "全场stint数": "—", "stint结构": branch_txt})
    rows.append({"车手": "—对照—", "红旗重启处换胎": f"实际换胎 {changed}/{total_drivers}",
                 "全场stint数": "—",
                 "stint结构": f"算法建议与多数车队一致: {agree}"})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(CHART_DIR, "18_历史回溯_2024红旗案例.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"已保存: {csv_path}")

    # ---- 图: 各车手 stint 时间线 + 红旗标记 + 模型最优策略
    fig, ax = plt.subplots(figsize=(13, 7))
    drivers_sorted = sorted(drivers,
                            key=lambda d: -stints[stints["Driver"] == d]["Laps"].sum())
    for i, drv in enumerate(drivers_sorted):
        sd = stints[stints["Driver"] == drv].sort_values("Stint")
        for r in sd.itertuples():
            ax.barh(i, r.EndLap - r.StartLap + 1, left=r.StartLap - 1,
                    color=COMP_COLORS.get(r.Compound, "#888"), alpha=0.85,
                    edgecolor="white", lw=0.5)
    ax.set_yticks(range(len(drivers_sorted)))
    ax.set_yticklabels(drivers_sorted, fontsize=8)
    ax.invert_yaxis()
    for rl in red_laps:
        ax.axvline(rl - 0.5, color="#e10600", lw=1.6, ls="--")
    if restart_lap:
        ax.axvline(restart_lap - 0.5, color="#3f8cff", lw=1.2, ls=":")
        ax.text(restart_lap, -1.2, f"红旗重启(L{restart_lap})", fontsize=9,
                color="#3f8cff")
    ax.set_xlabel("圈数")
    ax.set_title("2024 摩纳哥正赛 · 各车手实际轮胎策略时间线(色块=配方)\n"
                 f"红旗圈 L{red_laps} · 重启处换胎 {changed}/{total_drivers} 车手 · "
                 f"模型建议「{rec['label']}」", fontsize=11)
    ax.grid(alpha=0.25, ls=":", axis="x")
    fig.tight_layout()
    png_path = os.path.join(CHART_DIR, "18_历史回溯_2024红旗案例.png")
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"已保存: {png_path}")


if __name__ == "__main__":
    main()

