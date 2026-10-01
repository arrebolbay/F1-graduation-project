#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
赛前最优停站策略求解(DP动态规划)

状态: (圈数, 配方, 胎龄, 已用配方掩码)
决策: 继续跑 / 进站换胎
目标: 最小化总比赛时间
约束: 至少使用2种配方(FIA规则)

产出:
    08_最优停站策略时间线.png     Top3策略时间线
    09_策略累计时间对比.png       累计用时对比
    10_策略对比表.csv             DP vs 倍耐力 vs 实际
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

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
TOTAL_LAPS = 78                  # 摩纳哥正赛圈数
PIT_LOSS = 19                      # 进站损失(秒)(统一按用户指定)
COMPOUNDS = ["C5", "C4", "C3"]   # 软/中/硬
MAX_TYRE_AGE = 50                # 胎龄上限
MIN_STINT_LAPS = 5               # 每段最少圈数(防止最后一圈进站钻规则空子)
BASE_LAP_TIME = 75.0             # 100%性能时的基准圈速(秒)

# 性能曲线参数(从 fit_tire_degradation.py 拟合结果读取)
PERF_PARAMS = {
    "C5": {"P0": 97.7, "peak_val": 98.4, "t_peak": 1.0, "deg_rate": 0.114},
    "C4": {"P0": 98.0, "peak_val": 98.5, "t_peak": 0.0, "deg_rate": 0.027},
    "C3": {"P0": 98.3, "peak_val": 98.9, "t_peak": 18.0, "deg_rate": 0.026},
}

# 倍耐力官方策略(手动收集,格式: [(停站次数, [(起始配方, 圈数), ...]), ...])
PIRELLI_STRATEGIES = [
    (1, [("MEDIUM", 36), ("HARD", 42)]),
    (1, [("SOFT", 32), ("HARD", 46)]),
]


def perf_pct(t, params):
    """性能保持率(%)关于胎龄 t 的三阶段函数。"""
    P0 = params["P0"]
    peak_val = params["peak_val"]
    t_peak = params["t_peak"]
    deg_rate = params["deg_rate"]
    if t <= t_peak:
        return P0 + (peak_val - P0) * (t / max(t_peak, 1.0))
    else:
        return peak_val - deg_rate * (t - t_peak)


def lap_time(compound_idx, tyre_age, params=None, base_time=None):
    """给定配方和胎龄,返回圈速(秒)。base_time=None 时用全场基准。"""
    if params is None:
        params = PERF_PARAMS[COMPOUNDS[compound_idx]]
    if base_time is None:
        base_time = BASE_LAP_TIME
    perf = perf_pct(tyre_age, params)
    return base_time * 100.0 / max(perf, 50.0)


# ----------------------------------------------------------------- DP求解器
def solve_dp(start_compound_idx=0, perf_params=None, min_stops=1, base_time=None):
    """动态规划求解最优停站策略。

    min_stops: 最少进站次数(1=正常规则,2=强制两停)
    base_time: 基准圈速(秒),None 时用全场默认
    """
    if perf_params is None:
        perf_params = PERF_PARAMS
    n_comp = len(COMPOUNDS)
    INF = float("inf")

    dp = {}
    parent = {}
    # state = (lap, compound, tyre_age, used_mask, n_stops)
    start = (0, start_compound_idx, 0, 1 << start_compound_idx, 0)
    dp[start] = 0.0

    for lap in range(TOTAL_LAPS):
        current_states = [s for s in dp if s[0] == lap]
        for state in current_states:
            _, comp, age, mask, n_stops = state
            t = dp[state]
            cp = perf_params[COMPOUNDS[comp]]

            new_age = age + 1
            new_state = (lap + 1, comp, new_age, mask, n_stops)
            new_time = t + lap_time(comp, age, cp, base_time)
            if new_state not in dp or new_time < dp[new_state]:
                dp[new_state] = new_time
                parent[new_state] = state

            for new_comp in range(n_comp):
                if new_comp == comp:
                    continue
                if age < MIN_STINT_LAPS:
                    continue
                if TOTAL_LAPS - (lap + 1) < MIN_STINT_LAPS:
                    continue
                new_state = (lap + 1, new_comp, 0,
                             mask | (1 << new_comp), n_stops + 1)
                new_time = t + lap_time(comp, age, cp, base_time) + PIT_LOSS
                if new_state not in dp or new_time < dp[new_state]:
                    dp[new_state] = new_time
                    parent[new_state] = state

    # 找最优终态:至少 min_stops 次进站 + 至少 2 种配方
    best_time = INF
    best_state = None
    for state, t in dp.items():
        lap, comp, age, mask, n_stops = state
        if (lap == TOTAL_LAPS and n_stops >= min_stops
                and bin(mask).count("1") >= 2):
            if t < best_time:
                best_time = t
                best_state = state

    if best_state is None:
        return INF, []

    path = []
    state = best_state
    while state in parent:
        path.append(state)
        state = parent[state]
    path.append(state)
    path.reverse()

    return best_time, path


def extract_stints(path):
    """从策略路径提取停站信息。"""
    stints = []
    current_comp = path[0][1]
    start_lap = 0
    for i in range(1, len(path)):
        if path[i][1] != current_comp:
            stints.append({
                "compound": COMPOUNDS[current_comp],
                "start_lap": start_lap,
                "end_lap": path[i][0] - 1,
                "laps": path[i][0] - start_lap,
            })
            current_comp = path[i][1]
            start_lap = path[i][0]
    # 最后一段
    stints.append({
        "compound": COMPOUNDS[current_comp],
        "start_lap": start_lap,
        "end_lap": TOTAL_LAPS,
        "laps": TOTAL_LAPS - start_lap,
    })
    return stints


# ----------------------------------------------------------------- 按车手求解
def load_driver_params(csv_path):
    """加载按车手的衰减参数表(07b_车手衰减参数表.csv)。"""
    df = pd.read_csv(csv_path)
    driver_params = {}
    for _, row in df.iterrows():
        drv = row["车手"]
        comp = row["配方"]
        if drv not in driver_params:
            driver_params[drv] = {}
        driver_params[drv][comp] = {
            "P0": row["P0_pct"],
            "peak_val": row["peak_pct"],
            "t_peak": row["t_peak_laps"],
            "deg_rate": row["deg_rate_pct_per_lap"],
        }
    return driver_params


def solve_dp_driver(driver, driver_params, min_stops=1, base_time=None):
    """按车手的衰减曲线求解最优策略(尝试3种起步配方取最优)。"""
    pp = driver_params[driver]
    best_time, best_start, best_path = float("inf"), 0, []
    for start_idx in range(len(COMPOUNDS)):
        time, path = solve_dp(start_idx, pp, min_stops, base_time)
        if path and time < best_time:
            best_time, best_start, best_path = time, start_idx, path
    return best_time, best_start, best_path


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


# ----------------------------------------------------------------- 可视化
COMPOUND_COLORS = {"C5": "#e10600", "C4": "#ffd700", "C3": "#b0b0b0"}


def plot_strategy_timeline(results, out_path):
    """Top3策略时间线(色块图)。"""
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 2.8 * len(results)),
                             sharex=True)
    if len(results) == 1:
        axes = [axes]

    for idx, (total_time, start_idx, stints) in enumerate(results):
        ax = axes[idx]
        for stint in stints:
            color = COMPOUND_COLORS.get(stint["compound"], "#888")
            ax.barh(0, stint["laps"], left=stint["start_lap"],
                    height=0.5, color=color, edgecolor="black", linewidth=0.5)
            ax.text(stint["start_lap"] + stint["laps"] / 2, 0,
                    f"{stint['compound']}\n{stint['laps']}圈",
                    ha="center", va="center", fontsize=8, fontweight="bold")
        n_stops = len(stints) - 1
        total_str = f"{int(total_time // 3600)}:{int(total_time % 3600 // 60):02d}:{total_time % 60:04.1f}"
        ax.set_ylabel(f"策略{idx + 1}\n{n_stops}停", fontsize=10)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_title(f"策略{idx + 1}: {n_stops}停 | 总用时 {total_str}",
                     fontsize=10, loc="left")
        ax.grid(axis="x", alpha=0.3, ls=":")

    axes[-1].set_xlabel("圈数", fontsize=11)
    axes[-1].set_xlim(0, TOTAL_LAPS)
    fig.suptitle("摩纳哥站最优停站策略时间线(DP求解)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


def plot_cumulative_time(results, out_path):
    """累计用时对比曲线。"""
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = ["#e10600", "#1f6fb2", "#2ca25f"]

    for idx, (total_time, start_idx, stints) in enumerate(results):
        # 计算每圈的累计时间
        cum_times = [0.0]
        for stint in stints:
            comp_idx = COMPOUNDS.index(stint["compound"])
            for lap in range(stint["start_lap"], stint["end_lap"]):
                age = lap - stint["start_lap"]
                cum_times.append(cum_times[-1] + lap_time(comp_idx, age))
        laps = list(range(len(cum_times)))
        label = f"策略{idx + 1}: {'→'.join(s['compound'] for s in stints)}"
        ax.plot(laps, cum_times, lw=2, color=colors[idx % len(colors)],
                label=label)

    ax.set_xlabel("圈数", fontsize=12)
    ax.set_ylabel("累计用时 (秒)", fontsize=12)
    ax.set_title("不同停站策略的累计用时对比", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


# ----------------------------------------------------------------- Pirelli对比
def compare_with_pirelli(dp_results, out_path):
    """DP最优策略 vs 倍耐力官方策略 对比表。"""
    rows = []
    for idx, (total_time, start_idx, stints) in enumerate(dp_results):
        n_stops = len(stints) - 1
        stint_str = " → ".join(f"{s['compound']}({s['laps']}圈)" for s in stints)
        total_str = f"{int(total_time // 3600)}:{int(total_time % 3600 // 60):02d}:{total_time % 60:04.1f}"
        rows.append({
            "来源": f"DP策略{idx + 1}",
            "停站次数": n_stops,
            "策略": stint_str,
            "总用时": total_str,
            "总秒数": round(total_time, 1),
        })

    for idx, (n_stops, stint_list) in enumerate(PIRELLI_STRATEGIES):
        stint_str = " → ".join(f"{c}({l}圈)" for c, l in stint_list)
        rows.append({
            "来源": f"倍耐力推荐{idx + 1}",
            "停站次数": n_stops,
            "策略": stint_str,
            "总用时": "—",
            "总秒数": None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {os.path.basename(out_path)}")

    # 打印对比
    print("\n  ═══ 策略对比表 ═══")
    print(df.to_string(index=False))


def plot_driver_strategies(driver_strategies, out_path):
    """按车手最优策略时间线(每车手一行)。"""
    n = len(driver_strategies)
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for idx, (drv, info) in enumerate(driver_strategies.items()):
        ax = axes[idx]
        for stint in info["stints"]:
            color = COMPOUND_COLORS.get(stint["compound"], "#888")
            ax.barh(0, stint["laps"], left=stint["start_lap"],
                    height=0.6, color=color, edgecolor="black", linewidth=0.3)
        name = DRIVER_NAMES.get(drv, drv)
        ax.set_ylabel(f"{name}", fontsize=8, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.set_ylim(-0.5, 0.5)
        ax.grid(axis="x", alpha=0.2, ls=":")
    axes[-1].set_xlabel("圈数", fontsize=11)
    axes[-1].set_xlim(0, TOTAL_LAPS)
    fig.suptitle("各车手最优停站策略时间线(DP求解)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


def save_driver_strategy_table(driver_strategies, out_path):
    """按车手策略对比表。"""
    rows = []
    for drv, info in driver_strategies.items():
        stints = info["stints"]
        stint_str = " → ".join(f"{s['compound']}({s['laps']}圈)" for s in stints)
        tt = info["total_time"]
        total_str = f"{int(tt // 3600)}:{int(tt % 3600 // 60):02d}:{tt % 60:04.1f}"
        rows.append({
            "车手": drv,
            "中文名": DRIVER_NAMES.get(drv, drv),
            "停站次数": len(stints) - 1,
            "最优策略": stint_str,
            "总用时": total_str,
            "总秒数": round(tt, 1),
        })
    df = pd.DataFrame(rows).sort_values("总秒数")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {os.path.basename(out_path)}")
    print("\n  ═══ 各车手最优策略 ═══")
    print(df[["车手", "中文名", "停站次数", "最优策略", "总用时"]].to_string(index=False))


# ----------------------------------------------------------------- 主流程
def main():
    print("=" * 55)
    print(" 赛前最优停站策略求解(DP动态规划)")
    print("=" * 55)

    # 尝试3种起步配方,各求最优策略(正常规则:至少1停)
    results = []
    for start_idx in range(len(COMPOUNDS)):
        total_time, path = solve_dp(start_idx, min_stops=1)
        if path:
            stints = extract_stints(path)
            results.append((total_time, start_idx, stints))
            n_stops = len(stints) - 1
            stint_str = " → ".join(f"{s['compound']}({s['laps']}圈)" for s in stints)
            total_str = f"{int(total_time // 3600)}:{int(total_time % 3600 // 60):02d}:{total_time % 60:04.1f}"
            print(f"  起步{COMPOUNDS[start_idx]}: {n_stops}停 | {stint_str} | {total_str}")

    # 按总时间排序,取前3
    results.sort(key=lambda x: x[0])
    results = results[:3]

    print(f"\n  Top {len(results)} 策略已选出")

    # 输出
    print("\n[输出]")
    plot_strategy_timeline(results,
                           os.path.join(CHART_DIR, "08_最优停站策略时间线.png"))
    plot_cumulative_time(results,
                         os.path.join(CHART_DIR, "09_策略累计时间对比.png"))
    compare_with_pirelli(results,
                         os.path.join(CHART_DIR, "10_策略对比表.csv"))

    print("\n[按车手求解]")
    driver_csv = os.path.join(CHART_DIR, "07b_车手衰减参数表.csv")
    metrics_csv = os.path.join(CHART_DIR, "07c_车手驾驶习惯指标.csv")
    base_times = {}
    if os.path.exists(metrics_csv):
        mdf = pd.read_csv(metrics_csv)
        for _, row in mdf.iterrows():
            base_times[row["车手"]] = row["基准圈速_s"]
    if os.path.exists(driver_csv):
        driver_params = load_driver_params(driver_csv)
        driver_strategies = {}
        for drv in sorted(driver_params):
            bt = base_times.get(drv, BASE_LAP_TIME)
            tt, start_idx, path = solve_dp_driver(drv, driver_params, min_stops=1, base_time=bt)
            if path:
                driver_strategies[drv] = {
                    "total_time": tt,
                    "stints": extract_stints(path),
                }
        plot_driver_strategies(driver_strategies,
                               os.path.join(CHART_DIR, "08b_车手最优策略对比.png"))
        save_driver_strategy_table(driver_strategies,
                                   os.path.join(CHART_DIR, "10b_车手策略对比表.csv"))

        # 强制两停:按车手求解
        print("\n[按车手求解:强制两停]")
        driver_strategies_2stop = {}
        for drv in sorted(driver_params):
            bt = base_times.get(drv, BASE_LAP_TIME)
            tt2, _, path2 = solve_dp_driver(drv, driver_params, min_stops=2, base_time=bt)
            if path2:
                driver_strategies_2stop[drv] = {
                    "total_time": tt2,
                    "stints": extract_stints(path2),
                }
        if driver_strategies_2stop:
            plot_driver_strategies(driver_strategies_2stop,
                                   os.path.join(CHART_DIR, "08c_车手强制两停策略.png"))
            save_driver_strategy_table(driver_strategies_2stop,
                                       os.path.join(CHART_DIR, "10c_车手强制两停对比表.csv"))

        # 综合对比:正常 vs 强制两停
        print("\n[综合对比:正常 vs 强制两停]")
        rows = []
        for drv in sorted(driver_params):
            r1 = driver_strategies.get(drv)
            r2 = driver_strategies_2stop.get(drv)
            s1 = " → ".join(f"{s['compound']}({s['laps']}圈)" for s in r1["stints"]) if r1 else "—"
            s2 = " → ".join(f"{s['compound']}({s['laps']}圈)" for s in r2["stints"]) if r2 else "—"
            t1 = f"{r1['total_time']:.1f}" if r1 else "—"
            t2 = f"{r2['total_time']:.1f}" if r2 else "—"
            diff = f"{r2['total_time'] - r1['total_time']:+.1f}" if (r1 and r2) else "—"
            rows.append({
                "车手": drv, "中文名": DRIVER_NAMES.get(drv, drv),
                "正常策略": s1, "正常用时_s": t1,
                "两停策略": s2, "两停用时_s": t2, "两停损失_s": diff,
            })
        df_cmp = pd.DataFrame(rows)
        cmp_path = os.path.join(CHART_DIR, "10d_正常vs两停对比表.csv")
        df_cmp.to_csv(cmp_path, index=False, encoding="utf-8-sig")
        print(f"  已保存: {os.path.basename(cmp_path)}")
        print(df_cmp[["车手", "中文名", "正常策略", "两停策略", "两停损失_s"]].to_string(index=False))
    results_2stop = []
    for start_idx in range(len(COMPOUNDS)):
        total_time, path = solve_dp(start_idx, min_stops=2)
        if path:
            stints = extract_stints(path)
            results_2stop.append((total_time, start_idx, stints))
    results_2stop.sort(key=lambda x: x[0])
    if results_2stop:
        best = results_2stop[0]
        stint_str = " → ".join(f"{s['compound']}({s['laps']}圈)" for s in best[2])
        tt = best[0]
        total_str = f"{int(tt // 3600)}:{int(tt % 3600 // 60):02d}:{tt % 60:04.1f}"
        print(f"  强制两停最优: {stint_str} | {total_str}")
    else:
        print("  强制两停无可行解")

    print("\n完成!")


if __name__ == "__main__":
    main()



