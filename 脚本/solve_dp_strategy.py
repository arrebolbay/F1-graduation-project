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
import sys
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

# ----------------------------------------------------------------- 统一口径
# 全部常数 / 拟合参数 / 性能曲线 / 圈速公式 / DP 求解都来自 strategy_core
# (策略层单一事实来源)。2026-10 统一前,本文件硬编码了一份 PERF_PARAMS 且用
# ``base * 100 / perf``(未按峰值归一),与 web 端解出完全不同的"最优策略"
# (离线 C4→C3,网页 C5→C4) —— 与清洗口径分叉是同一类问题。
sys.path.insert(0, SCRIPT_DIR)
import strategy_core as core  # noqa: E402

TOTAL_LAPS = core.TOTAL_LAPS
PIT_LOSS = core.PIT_LOSS
COMPOUNDS = core.COMPOUNDS
MAX_TYRE_AGE = core.MAX_TYRE_AGE      # 仅参考上限,不进入约束(见 core 注释)
MIN_STINT_LAPS = core.MIN_STINT_LAPS
BASE_LAP_TIME = core.BASE_LAP_TIME    # 全场基准;按车手求解时用 07c 的基准圈速

# 全场拟合参数: 读 07_拟合参数表.csv(含暖胎段/负衰减截断),缺失才用兜底值
PERF_PARAMS = core.load_pooled_params()


# 倍耐力官方策略(手动收集,格式: [(停站次数, [(起始配方, 圈数), ...]), ...])
PIRELLI_STRATEGIES = [
    (1, [("MEDIUM", 36), ("HARD", 42)]),
    (1, [("SOFT", 32), ("HARD", 46)]),
]


def perf_pct(t, params):
    """性能保持率(%)(保留函数名以兼容旧调用,实现见 strategy_core)。"""
    return core.perf_pct(t, core.normalize_params(params))


def lap_time(compound, tyre_age, params=None, base_time=None):
    """单圈圈速(秒): ``(基准 + 配方偏移) / (perf/peak)``,与 web 端同一公式。

    ``compound`` 可为配方下标(旧调用习惯)或配方名。
    """
    comp = COMPOUNDS[compound] if isinstance(compound, int) else compound
    return core.lap_time(comp, tyre_age, base_time or BASE_LAP_TIME,
                         params or PERF_PARAMS[comp])


# ----------------------------------------------------------------- DP求解器
def solve_dp(start_compound_idx=0, perf_params=None, min_stops=1, base_time=None,
             max_stops=None):
    """动态规划求解(实现见 strategy_core.solve_dp)。

    max_stops=1 → 与 web 的 normal 模式(恰好 1 停)对齐;
    max_stops=None → 纯时间最优(不限停站次数)。
    """
    return core.solve_dp(start_compound_idx, perf_params or PERF_PARAMS,
                         min_stops=min_stops, base_time=base_time,
                         max_stops=max_stops)


def extract_stints(path):
    """从策略路径提取停站信息(闭区间 start/end,laps = end - start + 1)。"""
    return core.extract_stints(path, total_laps=TOTAL_LAPS)



# ----------------------------------------------------------------- 按车手求解
def load_driver_params(csv_path):
    """加载按车手的衰减参数表(07b) —— 读取 + 截断口径见 strategy_core。"""
    return core.load_driver_params(os.path.dirname(os.path.abspath(csv_path)))


def solve_dp_driver(driver, driver_params, min_stops=1, base_time=None,
                    max_stops=None):
    """按车手的衰减曲线求解最优策略(三种起步配方取最优)。

    车手缺失的配方回退到全场汇总(``core.driver_compound_params``),
    避免 07b 里某配方样本不足时直接 KeyError。
    """
    pp = core.driver_compound_params(driver_params, PERF_PARAMS, driver)
    return core.solve_dp_best_start(pp, min_stops=min_stops,
                                    base_time=base_time, max_stops=max_stops)



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


def plot_cumulative_time(results, out_path, base_time=None, params=None):
    """累计用时对比曲线。

    逐圈累计由 :func:`strategy_core.cumulative_curve` 计算 —— 该函数与 DP 目标
    函数同源,进站损失计入新 stint 首圈,末值必须等于 DP 总用时。

    旧版两个 bug(2026-10 修复):
      1. ``range(start_lap, end_lap)`` 把闭区间当开区间,每段少画 1 圈
         (整条曲线少 停站次数 圈,总圈数 ≠ 78);
      2. 重算时没传 params / base_time,用的是硬编码默认参数 ——
         图上累计时间与表里的总用时不是同一次计算。
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = ["#e10600", "#1f6fb2", "#2ca25f"]
    params = params or PERF_PARAMS

    for idx, (_total_time, _start_idx, stints) in enumerate(results):
        laps, cum = core.cumulative_curve(stints, base_time, params)
        label = (f"策略{idx + 1}: {'→'.join(s['compound'] for s in stints)}"
                 f"(累计 {cum[-1]:.1f}s)")
        ax.plot(laps, cum, lw=2, color=colors[idx % len(colors)], label=label)
        for st in stints[1:]:
            ax.axvline(st["start_lap"], color=colors[idx % len(colors)],
                       lw=0.8, ls=":", alpha=0.6)

    ax.set_xlim(0, TOTAL_LAPS)
    ax.set_xlabel("圈数", fontsize=12)
    ax.set_ylabel("累计用时 (秒)", fontsize=12)
    ax.set_title("不同停站策略的累计用时对比(竖虚线=进站圈)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


# ----------------------------------------------------------------- Pirelli对比
def compare_with_pirelli(dp_results, out_path, unrestricted=None):
    """DP最优策略 vs 倍耐力官方策略 对比表。

    ``unrestricted=(总秒数, stints)``: 不限停站次数(纯时间最优)的参考行 ——
    用于说明"模型在 78 圈 / 19 秒进站损失下认为 2 停只比 1 停快零点几秒,
    因而摩纳哥实践中普遍选择 1 停保位置"(纯时间模型不含赛道位置与超车难度)。
    """
    rows = []
    for idx, (total_time, start_idx, stints) in enumerate(dp_results):
        n_stops = len(stints) - 1
        stint_str = " → ".join(f"{s['compound']}({s['laps']}圈)" for s in stints)
        total_str = f"{int(total_time // 3600)}:{int(total_time % 3600 // 60):02d}:{total_time % 60:04.1f}"
        rows.append({
            "来源": f"DP策略{idx + 1}(1停基准)",
            "停站次数": n_stops,
            "策略": stint_str,
            "总用时": total_str,
            "总秒数": round(total_time, 1),
        })

    if unrestricted:
        u_total, u_stints = unrestricted
        u_str = " → ".join(f"{s['compound']}({s['laps']}圈)" for s in u_stints)
        rows.append({
            "来源": "DP(不限停站·纯时间最优)",
            "停站次数": len(u_stints) - 1,
            "策略": u_str,
            "总用时": f"{int(u_total // 3600)}:{int(u_total % 3600 // 60):02d}:{u_total % 60:04.1f}",
            "总秒数": round(u_total, 1),
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

    # 尝试3种起步配方,各求最优策略
    # max_stops=1 → "恰好 1 停",与 web 端 normal 模式(1 停基准)完全对齐;
    # 不限停站次数的纯时间最优另算一次,作为 10 表的参考行。
    results = []
    for start_idx in range(len(COMPOUNDS)):
        total_time, path = solve_dp(start_idx, min_stops=1, max_stops=1)
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

    u_total, _u_idx, u_path = core.solve_dp_best_start(PERF_PARAMS, min_stops=1)
    unrestricted = (u_total, extract_stints(u_path)) if u_path else None
    if unrestricted:
        print(f"  参考(不限停站): "
              f"{core.format_strategy(unrestricted[1])} | {u_total:.1f}s")


    print(f"\n  Top {len(results)} 策略已选出")

    # 输出
    print("\n[输出]")
    plot_strategy_timeline(results,
                           os.path.join(CHART_DIR, "08_最优停站策略时间线.png"))
    plot_cumulative_time(results,
                         os.path.join(CHART_DIR, "09_策略累计时间对比.png"))
    compare_with_pirelli(results,
                         os.path.join(CHART_DIR, "10_策略对比表.csv"),
                         unrestricted=unrestricted)

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
            tt, start_idx, path = solve_dp_driver(drv, driver_params,
                                                  min_stops=1, base_time=bt,
                                                  max_stops=1)
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
            tt2, _, path2 = solve_dp_driver(drv, driver_params, min_stops=2,
                                            base_time=bt, max_stops=2)
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
        total_time, path = solve_dp(start_idx, min_stops=2, max_stops=2)
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



