#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
赛中Stackelberg博弈决策引擎(问题二核心)

三种场景:
  1. Undercut: 你=领导者(先进站), 对手=跟随者(跟不跟)
  2. Overcut:  对手=领导者(先进站), 你=跟随者(跟不跟)
  3. SC/VSC:   是否利用廉价进站机会

输出:
    11_Stackelberg决策结果.csv     三场景分成功率
    12_关键拐点.png              换胎时间 vs Undercut成功率
    13_轮胎选择.png              不同配方期望完赛时间
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CHART_DIR = os.path.join(PROJECT_DIR, "图表")
os.makedirs(CHART_DIR, exist_ok=True)

for _n in ("Microsoft YaHei", "SimHei", "SimSun"):
    try:
        fm.findfont(_n, fallback_to_default=False)
    except Exception:
        continue
    plt.rcParams["font.sans-serif"] = [_n, "DejaVu Sans"]
    break
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------- 参数
TOTAL_LAPS = 78
PIT_LOSS = 19                     # 正常进站损失(秒)
PIT_LOSS_SC = PIT_LOSS * 0.35     # SC下进站损失(约 6-7 秒)
PIT_LOSS_VSC = PIT_LOSS * 0.55    # VSC下进站损失(约 10 秒)
COMPOUNDS = ["C5", "C4", "C3"]
N_SIM = 10000                     # 蒙特卡洛模拟次数

# 轮胎衰减参数(全场统一,可由 07_拟合参数表 更新)
PERF_PARAMS = {
    "C5": {"P0": 97.7, "peak_val": 98.4, "t_peak": 1.0, "deg_rate": 0.114},
    "C4": {"P0": 98.0, "peak_val": 98.5, "t_peak": 0.0, "deg_rate": 0.027},
    "C3": {"P0": 98.3, "peak_val": 98.9, "t_peak": 18.0, "deg_rate": 0.026},
}

# 换胎时间分布
PIT_NORMAL_PROB = 0.85
PIT_NORMAL_MEAN, PIT_NORMAL_STD = 2.5, 0.3
PIT_ABNORMAL_MEAN, PIT_ABNORMAL_STD = 5.0, 2.0
PIT_LANE_TRANSIT = 16.5          # 维修区通道行驶时间(秒,正常)
PIT_LANE_TRANSIT_SC = 4.0        # SC下通道行驶时间(限速更低)
PIT_LANE_TRANSIT_VSC = 8.0       # VSC下通道行驶时间

# 雨胎参数(估计值,无真实数据)
WET_PARAMS = {
    "INTER": {"P0": 96.0, "peak_val": 97.5, "t_peak": 2.0, "deg_rate": 0.15},
    "WET":   {"P0": 94.0, "peak_val": 96.0, "t_peak": 3.0, "deg_rate": 0.25},
}

DRIVER_NAMES = {
    "ALB": "阿尔本", "ALO": "阿隆索", "ANT": "安东内利",
    "BEA": "比尔曼", "BOR": "博尔托莱托", "BOT": "博塔斯",
    "COL": "科拉皮托", "DEV": "德弗里斯", "DOO": "杜汉",
    "GAS": "加斯利", "HAD": "哈贾尔", "HAM": "汉密尔顿",
    "HUL": "胡肯伯格", "LAT": "拉提菲", "LAW": "劳森",
    "LEC": "勒克莱尔", "MAG": "马格努森", "MSC": "米克·舒马赫",
    "NOR": "诺里斯", "OCO": "奥康", "PER": "佩雷兹",
    "PIA": "皮亚斯特里", "RIC": "里卡多", "RUS": "拉塞尔",
    "SAI": "塞恩斯", "SAR": "萨金特", "STR": "斯特罗尔",
    "TSU": "角田裕毅", "VER": "维斯塔潘", "VET": "维特尔",
    "ZHO": "周冠宇",
}

COMPOUND_COLORS = {"C5": "#e10600", "C4": "#ffd700", "C3": "#b0b0b0"}

# 不同配方的绝对速度偏移(秒,相对于 C4=0,摩纳哥适用)
COMPOUND_OFFSET = {"C3": 1.0, "C4": 0.0, "C5": -0.5}


# ----------------------------------------------------------------- 核心函数
def perf_pct(t, params):
    P0, peak_val, t_peak, deg_rate = (params["P0"], params["peak_val"],
                                       params["t_peak"], params["deg_rate"])
    if t <= t_peak:
        return P0 + (peak_val - P0) * (t / max(t_peak, 1))
    return peak_val - deg_rate * (t - t_peak)


def lap_time(compound, tyre_age, base_time=75.0, params=None):
    if params is None:
        params = PERF_PARAMS.get(compound, PERF_PARAMS["C4"])
    p = perf_pct(tyre_age, params)
    offset = COMPOUND_OFFSET.get(compound, 0.0)
    return (base_time + offset) * 100.0 / max(p, 50.0)


def sample_pit_time(pit_lane_transit=None):
    """抽样总进站损失(秒) = 通道行驶(参数) + 换胎(随机)。"""
    if pit_lane_transit is None:
        pit_lane_transit = PIT_LANE_TRANSIT
    if np.random.random() < PIT_NORMAL_PROB:
        change = max(1.5, np.random.normal(PIT_NORMAL_MEAN, PIT_NORMAL_STD))
    else:
        change = max(3.0, np.random.normal(PIT_ABNORMAL_MEAN, PIT_ABNORMAL_STD))
    return pit_lane_transit + change


# ----------------------------------------------------------------- 场景一:Undercut
def simulate_undercut(my_comp, my_age, rival_comp, rival_age,
                      gap_s, remaining, my_base=75.0, rival_base=75.0):
    """你=领导者(先进站)。返回两种分支的成功率。"""
    wins_follow = 0
    wins_stay = 0
    for _ in range(N_SIM):
        # A:对手跟进(双方都进站)
        t_me = sample_pit_time()
        t_rival = sample_pit_time() + gap_s
        a_me, a_riv = 0, 0
        for _ in range(remaining):
            t_me += lap_time(my_comp, a_me, my_base)
            t_rival += lap_time(rival_comp, a_riv, rival_base)
            a_me += 1; a_riv += 1
        wins_follow += int(t_me < t_rival)

        # B:对手不跟(只有你进站)
        t_me = sample_pit_time()
        t_rival = gap_s
        a_me, a_riv = 0, rival_age
        for _ in range(remaining):
            t_me += lap_time(my_comp, a_me, my_base)
            t_rival += lap_time(rival_comp, a_riv, rival_base)
            a_me += 1; a_riv += 1
        wins_stay += int(t_me < t_rival)

    return {"success_follow": wins_follow / N_SIM * 100,
            "success_stay": wins_stay / N_SIM * 100}


# ----------------------------------------------------------------- 场景二:Overcut
def simulate_overcut(my_comp, my_age, rival_comp, rival_age,
                     gap_s, remaining, my_base=75.0, rival_base=75.0):
    """对手=领导者(先进站)。返回两种分支的成功率。"""
    wins_stay = 0
    wins_follow = 0
    for _ in range(N_SIM):
        # A:你留在赛道
        t_me = 0.0; t_rival = sample_pit_time() + gap_s
        a_me, a_riv = my_age, 0
        for _ in range(remaining):
            t_me += lap_time(my_comp, a_me, my_base)
            t_rival += lap_time(rival_comp, a_riv, rival_base)
            a_me += 1; a_riv += 1
        wins_stay += int(t_me < t_rival)

        # B:你跟进进站
        t_me = sample_pit_time(); t_rival = sample_pit_time() + gap_s
        a_me, a_riv = 0, 0
        for _ in range(remaining):
            t_me += lap_time(my_comp, a_me, my_base)
            t_rival += lap_time(rival_comp, a_riv, rival_base)
            a_me += 1; a_riv += 1
        wins_follow += int(t_me < t_rival)

    return {"success_stay": wins_stay / N_SIM * 100,
            "success_follow": wins_follow / N_SIM * 100}


# ----------------------------------------------------------------- 场景三:SC/VSC
def simulate_sc(my_comp, my_age, rival_comp, rival_age,
                gap_s, remaining, sc_type="SC",
                my_base=75.0, rival_base=75.0):
    """SC/VSC 进站决策。sc_type: "SC" 或 "VSC"。"""
    pit_transit = PIT_LANE_TRANSIT_SC if sc_type == "SC" else PIT_LANE_TRANSIT_VSC
    wins_pit = 0; wins_stay = 0
    for _ in range(N_SIM):
        # A:你进站(廉价)
        t_me = sample_pit_time(pit_transit); t_rival = gap_s
        a_me, a_riv = 0, rival_age
        for _ in range(remaining):
            t_me += lap_time(my_comp, a_me, my_base)
            t_rival += lap_time(rival_comp, a_riv, rival_base)
            a_me += 1; a_riv += 1
        wins_pit += int(t_me < t_rival)

        # B:你不进站
        t_me = 0.0; t_rival = gap_s
        a_me, a_riv = my_age, rival_age
        for _ in range(remaining):
            t_me += lap_time(my_comp, a_me, my_base)
            t_rival += lap_time(rival_comp, a_riv, rival_base)
            a_me += 1; a_riv += 1
        wins_stay += int(t_me < t_rival)

    return {"success_pit": wins_pit / N_SIM * 100,
            "success_stay": wins_stay / N_SIM * 100}


# ----------------------------------------------------------------- 轮胎选择
def tire_choice_etime(compound_options, remaining, base_time=75.0):
    """计算每种配方的期望完赛时间(秒)。"""
    results = {}
    for comp in compound_options:
        params = PERF_PARAMS.get(comp, PERF_PARAMS["C4"])
        total = sum(lap_time(comp, a, base_time, params)
                    for a in range(remaining))
        results[comp] = total
    return results


# ----------------------------------------------------------------- 关键拐点
def find_critical_pit_time(my_comp, my_age, rival_comp, rival_age,
                           gap_s, remaining, my_base=75.0, rival_base=75.0,
                           threshold=50.0):
    """Undercut 成功率降到 threshold% 以下时的换胎时间(秒)。"""
    for pit_s in np.arange(15, 35, 0.5):
        wins = 0
        for _ in range(2000):
            t_me = pit_s; t_rival = gap_s
            a_me, a_riv = 0, rival_age
            for _ in range(remaining):
                t_me += lap_time(my_comp, a_me, my_base)
                t_rival += lap_time(rival_comp, a_riv, rival_base)
                a_me += 1; a_riv += 1
            wins += int(t_me < t_rival)
        if wins / 2000 * 100 < threshold:
            return pit_s
    return None


# ----------------------------------------------------------------- 可视化
def plot_decision_comparison(results, out_path):
    """三种场景分成功率对比柱状图。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Undercut
    ax = axes[0]
    labels = ["对手跟进", "对手不跟"]
    vals = [results["undercut"]["success_follow"],
            results["undercut"]["success_stay"]]
    bars = ax.bar(labels, vals, color=["#1f6fb2", "#2ca25f"], width=0.5)
    ax.set_ylim(0, 100); ax.set_ylabel("成功率 (%)")
    ax.set_title("Undercut (你先进站)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, ls=":")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Overcut
    ax = axes[1]
    labels = ["留在赛道", "跟进进站"]
    vals = [results["overcut"]["success_stay"],
            results["overcut"]["success_follow"]]
    bars = ax.bar(labels, vals, color=["#2ca25f", "#1f6fb2"], width=0.5)
    ax.set_ylim(0, 100)
    ax.set_title("Overcut (对手先进站)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, ls=":")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # SC
    ax = axes[2]
    labels = ["进站", "不进站"]
    vals = [results["sc"]["success_pit"], results["sc"]["success_stay"]]
    bars = ax.bar(labels, vals, color=["#e10600", "#4c72b0"], width=0.5)
    ax.set_ylim(0, 100); ax.set_title("安全车下决策", fontsize=11)
    ax.grid(axis="y", alpha=0.3, ls=":")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Stackelberg 博弈决策结果(分场景成功率)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


def plot_tire_choice(results, remaining, out_path):
    """不同配方期望完赛时间对比。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    comps = list(results.keys())
    times = [results[c] / 60 for c in comps]
    colors = [COMPOUND_COLORS.get(c, "#888") for c in comps]
    bars = ax.bar(comps, times, color=colors, width=0.5,
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("期望完赛时间 (分钟)")
    ax.set_title(f"轮胎配方选择(剩余 {remaining} 圈)", fontsize=13)
    ax.grid(axis="y", alpha=0.3, ls=":")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, t + 0.05,
                f"{t:.2f}min", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


def plot_critical_point(critical_pit, out_path):
    """关键拐点标注图。"""
    pit_times = np.arange(15, 35, 0.5)
    success_rates = []
    for pt in pit_times:
        wins = 0
        for _ in range(2000):
            t_me = pt; t_rival = 0
            a_me, a_riv = 0, 15
            for _ in range(30):
                t_me += lap_time("C4", a_me)
                t_rival += lap_time("C3", a_riv)
                a_me += 1; a_riv += 1
            wins += int(t_me < t_rival)
        success_rates.append(wins / 2000 * 100)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pit_times, success_rates, lw=2, color="#1f6fb2")
    ax.axhline(50, color="gray", ls=":", lw=1)
    if critical_pit:
        ax.axvline(critical_pit, color="#e10600", ls="--", lw=1.5,
                   label=f"临界换胎时间: {critical_pit:.1f}秒")
        ax.legend(fontsize=10)
    ax.set_xlabel("换胎时间 (秒)")
    ax.set_ylabel("Undercut 成功率 (%)")
    ax.set_title("关键拐点: 换胎时间 vs Undercut 成功率", fontsize=13)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


# ----------------------------------------------------------------- 主程序
def main():
    print("=" * 55)
    print(" 赛中Stackelberg博弈决策引擎")
    print("=" * 55)
    params = {
        "my_comp": "C4", "my_age": 22,
        "rival_comp": "C4", "rival_age": 12,
        "gap_s": 2.0,
        "remaining": TOTAL_LAPS - 45,
        "my_base": 74.5, "rival_base": 74.3,
    }
    print(f"\n  当前状态(示例: 第45圈,安全车刚出动)")
    print(f"  我:   {params['my_comp']} (胎龄{params['my_age']}圈)")
    print(f"  对手: {params['rival_comp']} (胎龄{params['rival_age']}圈)")
    print(f"  差距: {params['gap_s']}秒 | 剩余: {params['remaining']}圈")

    print("\n[场景一:Undercut]")
    undercut = simulate_undercut(**params)
    print(f"  对手跟进: {undercut['success_follow']:.1f}%")
    print(f"  对手不跟: {undercut['success_stay']:.1f}%")

    print("\n[场景二:Overcut]")
    overcut = simulate_overcut(**params)
    print(f"  留在赛道: {overcut['success_stay']:.1f}%")
    print(f"  跟进进站: {overcut['success_follow']:.1f}%")

    print("\n[场景三:SC/VSC]")
    sc = simulate_sc(**params, sc_type="SC")
    print(f"  SC进站: {sc['success_pit']:.1f}% | 不进站: {sc['success_stay']:.1f}%")
    vsc = simulate_sc(**params, sc_type="VSC")
    print(f"  VSC进站:{vsc['success_pit']:.1f}% | 不进站: {vsc['success_stay']:.1f}%")

    print("\n[轮胎选择]")
    tire_times = tire_choice_etime(["C5","C4","C3"],
                                   params["remaining"], params["my_base"])
    for c, t in sorted(tire_times.items(), key=lambda x: x[1]):
        print(f"  {c}: {t/60:.2f} 分钟")

    print("\n[关键拐点]")
    crit = find_critical_pit_time(**params)
    if crit:
        print(f"  换胎时间 > {crit:.1f}秒 → Undercut 降至50%以下")
    else:
        print("  未找到临界点")

    print("\n[输出]")
    all_results = {"undercut": undercut, "overcut": overcut, "sc": sc}
    plot_decision_comparison(all_results,
                             os.path.join(CHART_DIR, "11_Stackelberg决策结果.png"))
    plot_tire_choice(tire_times, params["remaining"],
                     os.path.join(CHART_DIR, "13_轮胎选择.png"))
    plot_critical_point(crit, os.path.join(CHART_DIR, "12_关键拐点.png"))

    summary = pd.DataFrame([
        {"场景": "Undercut(对手跟进)", "成功率%": round(undercut["success_follow"], 1)},
        {"场景": "Undercut(对手不跟)", "成功率%": round(undercut["success_stay"], 1)},
        {"场景": "Overcut(留在赛道)", "成功率%": round(overcut["success_stay"], 1)},
        {"场景": "Overcut(跟进进站)", "成功率%": round(overcut["success_follow"], 1)},
        {"场景": "SC进站", "成功率%": round(sc["success_pit"], 1)},
        {"场景": "SC不进站", "成功率%": round(sc["success_stay"], 1)},
        {"场景": "VSC进站", "成功率%": round(vsc["success_pit"], 1)},
        {"场景": "VSC不进站", "成功率%": round(vsc["success_stay"], 1)},
    ])
    summary.to_csv(os.path.join(CHART_DIR, "11_Stackelberg决策结果.csv"),
                   index=False, encoding="utf-8-sig")
    print("  已保存: 11_Stackelberg决策结果.csv")
    print("\n完成!")


if __name__ == "__main__":
    main()