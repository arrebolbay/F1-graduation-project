#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
策略层单一事实来源(Strategy Core) —— 轮胎性能曲线 + 圈速公式 + DP 求解

为什么要有这个文件
------------------
2026-10 统一清洗口径时发现同一个病在策略层还有一份:**同一份 07/07b/07c,
三个脚本给出三个最优策略**。

    脚本\solve_dp_strategy.py   硬编码 PERF_PARAMS(已与拟合结果脱钩)
    脚本\solve_stackelberg.py   另一份硬编码 PERF_PARAMS
    web\engine.py               读 CSV + 按 peak 归一 + 配方偏移

其中最关键的一条是圈速公式:旧写法 ``base * 100 / perf`` **没有按各自峰值归一**,
会把"配方基准性能水平的拟合差异"和"配方速度偏移"重复计入 —— engine.py 里已经修过
(注释原话:"曾导致 C3 被算成最快配方"),但两个离线脚本没跟上,于是一份数据两种结论。

本模块把下面这些东西收拢成唯一实现,四个调用方全部 import 它:

    * 常数(TOTAL_LAPS / PIT_LOSS / MIN_STINT_LAPS / COMPOUND_OFFSET ...)
    * 参数读取(07 全场汇总 / 07b 车手级 / 07c 基准圈速)
    * 物理合理性截断(:func:`sanitize_params`)
    * 性能曲线 :func:`perf_pct`、单圈圈速 :func:`lap_time`(按 peak 归一)
    * 离线 DP 求解 :func:`solve_dp`、分段提取 :func:`extract_stints`
    * 累计用时曲线 :func:`cumulative_curve`(逐圈含进站损失,修正原图少画一圈的 bug)

调用方
------
    web/engine.py                 轮胎模型 + DataStore 参数加载
    脚本/solve_dp_strategy.py     08/09/10/10b/10c/10d
    脚本/solve_stackelberg.py     11/12/13 + 收益矩阵
    脚本/validate_cross_year.py   15(仍走 engine 的配方库存 DP)

与 engine 的关系
----------------
engine 的 ``_solve_inventory`` 额外带"练习赛/排位赛轮胎库存"约束(用户可在页面里
选择自己还剩哪些胎),这是**合法的建模差异**,不在本模块重复实现;
但两者用的性能曲线、圈速公式、参数来源完全相同 —— 差异只来自"可用轮胎集合"。
"""

from __future__ import annotations

import os

import numpy as np

# ----------------------------------------------------------------- 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CHART_DIR = os.path.join(PROJECT_DIR, "图表")

# ----------------------------------------------------------------- 常数
TOTAL_LAPS = 78                     # 摩纳哥正赛圈数
COMPOUNDS = ["C5", "C4", "C3"]      # 软 / 中 / 硬(C6 为 2025 年摩纳哥软胎,不参与)
PIT_LOSS = 19.0                     # 正常进站总损失(秒) = 通道行驶 + 换胎
PIT_LOSS_SC = PIT_LOSS * 0.35       # 安全车下进站损失(约 6.7 秒)
PIT_LOSS_VSC = PIT_LOSS * 0.55      # 虚拟安全车下进站损失(约 10.5 秒)
MIN_STINT_LAPS = 5                  # 每段最少圈数(防止"最后一圈进站"钻规则空子)
BASE_LAP_TIME = 75.0                # 100% 性能时的全场基准圈速(秒)

# 单套轮胎最大使用圈数上限(DP 约束)。
# 取 TOTAL_LAPS = 等价于"不设人为上限":Pirelli 并未限制正赛里单套轮胎的使用圈数,
# 2024 摩纳哥红旗后多数车手用一套硬胎跑完 70+ 圈即是反例。旧 engine 里写死 50 圈,
# 会把"1 停跑 59 圈的 C4"判成不可行 → 网页解出 C5(28)→C4(50),而离线(无此上限)
# 解出 C4(59)→C5(19) —— 同一个车手两个"最优策略"的另一半原因。
MAX_TYRE_AGE = TOTAL_LAPS


# 不同配方的绝对速度偏移(秒,相对 C4=0,摩纳哥): 软胎更快、硬胎更慢。
# 配方速度差由 offset 承担,perf 曲线只负责"暖胎 + 衰减"形状(见 lap_time)。
COMPOUND_OFFSET = {"C3": 1.0, "C4": 0.0, "C5": -0.5}

# 07 表缺失时的兜底值(2022-2025 摩纳哥拟合结果,仅用于 CSV 读不到时防止崩溃)
FALLBACK_POOLED = {
    "C5": {"P0": 98.17, "peak": 98.55, "t_peak": 0.0, "deg": 0.1163},
    "C4": {"P0": 98.10, "peak": 98.57, "t_peak": 0.0, "deg": 0.0268},
    "C3": {"P0": 97.98, "peak": 98.91, "t_peak": 16.0, "deg": 0.0222},
}

# 参数键的兼容别名: 07 系列 CSV 用 peak_pct/deg_rate_pct_per_lap,
# 早期脚本用 peak_val/deg_rate,统一到 P0/peak/t_peak/deg。
_ALIASES = {
    "peak": ("peak", "peak_val", "peak_pct"),
    "deg": ("deg", "deg_rate", "deg_rate_pct_per_lap"),
    "P0": ("P0", "P0_pct"),
    "t_peak": ("t_peak", "t_peak_laps"),
}


def _csv_num(row, *names, default=0.0):
    """从一行 CSV(字典)里按候选列名取浮点数。"""
    for name in names:
        if name in row:
            try:
                v = float(row[name])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                return v
    return float(default)


def _read_csv_dicts(path):
    """读 CSV 为字典列表(不依赖 pandas,便于 engine 复用)。"""
    import csv

    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ============================================================================
# 一、参数读取(07 / 07b / 07c)
# ============================================================================
def normalize_params(raw):
    """把任意来源的参数字典统一成 ``{"P0","peak","t_peak","deg"}``。"""
    out = {}
    for key, names in _ALIASES.items():
        for name in names:
            if name in raw and raw[name] is not None:
                try:
                    out[key] = float(raw[name])
                except (TypeError, ValueError):
                    pass
                break
    out.setdefault("P0", 98.0)
    out.setdefault("peak", max(out["P0"], 98.5))
    out.setdefault("t_peak", 0.0)
    out.setdefault("deg", 0.03)
    return out


def sanitize_params(params):
    """暖胎段 + 衰减段的物理合理性截断(原 web/engine.py 的 ``_sanitize_deg``)。

    拟合的 P0→peak 爬升段混入了发车 / 交通等混杂效应(个别车手-配方组合出现
    t_peak 高达 22 圈、新胎比峰值慢 3~4 个性能点 → "新硬胎首圈竟慢 3 秒、
    轮胎越跑越快 20 余圈"的非物理结果)。真实暖胎约 1~3 圈、落差 ≤1.2 个性能点,
    故统一截断: ``t_peak ≤ 3 圈``、``P0 ≥ peak − 1.2``、``deg ≥ 0``。

    旧版只在 web 端截断、离线 DP 不截断 → 同一车手网页与脚本能解出不同策略,
    现由本函数统一(offline 与 engine 完全同源)。
    """
    p = normalize_params(params)
    peak = float(p["peak"])
    p["t_peak"] = min(max(float(p["t_peak"]), 0.0), 3.0)
    p["P0"] = max(float(p["P0"]), peak - 1.2)
    p["deg"] = max(float(p["deg"]), 0.0)          # 不允许"轮胎越跑越快"
    if p["peak"] < p["P0"]:
        p["peak"] = p["P0"]
    return p


def load_pooled_params(chart_dir=None, sanitize=True):
    """读 07_拟合参数表.csv → ``{配方: 参数}``;缺失的配方用兜底值。"""
    base = chart_dir or CHART_DIR
    out = {c: dict(v) for c, v in FALLBACK_POOLED.items()}
    for row in _read_csv_dicts(os.path.join(base, "07_拟合参数表.csv")):
        comp = (row.get("配方") or "").strip()
        if comp not in COMPOUNDS:
            continue
        out[comp] = normalize_params(row)
    return {c: (sanitize_params(p) if sanitize else p) for c, p in out.items()}


def load_driver_params(chart_dir=None, sanitize=True):
    """读 07b_车手衰减参数表.csv → ``{车手: {配方: 参数}}``。

    额外带回 ``来源``(driver / pooled)与 ``n``(样本数)供调用方判断可信度。
    """
    base = chart_dir or CHART_DIR
    out = {}
    for row in _read_csv_dicts(os.path.join(base, "07b_车手衰减参数表.csv")):
        code = (row.get("车手") or "").strip().upper()
        comp = (row.get("配方") or "").strip()
        if not code or comp not in COMPOUNDS:
            continue
        p = normalize_params(row)
        p["source"] = (row.get("来源") or "").strip()
        p["n"] = int(_csv_num(row, "样本数", default=0))
        out.setdefault(code, {})[comp] = sanitize_params(p) if sanitize else p
    return out


def load_base_times(chart_dir=None):
    """读 07c_车手驾驶习惯指标.csv → ``{车手: 基准圈速_s}``。"""
    base = chart_dir or CHART_DIR
    out = {}
    for row in _read_csv_dicts(os.path.join(base, "07c_车手驾驶习惯指标.csv")):
        code = (row.get("车手") or "").strip().upper()
        if not code:
            continue
        v = _csv_num(row, "基准圈速_s", default=0.0)
        if v > 0:
            out[code] = v
    return out


def driver_compound_params(driver_params, pooled_params, code):
    """车手级参数优先、缺配方回退全场汇总(保证三个配方都有值)。"""
    own = driver_params.get((code or "").upper(), {})
    return {c: dict(own.get(c) or pooled_params[c]) for c in COMPOUNDS}


# ============================================================================
# 二、轮胎性能模型(唯一实现)
# ============================================================================
def perf_pct(t, params):
    """性能保持率(%)关于胎龄 t: 暖胎线性上升 → 峰值 → 线性衰减。"""
    P0 = float(params["P0"])
    peak = float(params["peak"])
    t_peak = float(params["t_peak"])
    deg = float(params["deg"])
    if t <= t_peak:
        return P0 + (peak - P0) * (t / max(t_peak, 1.0))
    return peak - deg * (t - t_peak)


def lap_time(compound, tyre_age, base_time=None, params=None, offsets=None):
    """单圈圈速(秒) = (基准圈速 + 配方偏移) ÷ 归一化性能保持率。

    perf 曲线按各自峰值归一(``p / peak``,1.0 = 峰值状态):

      * 消除跨配方的"拟合水平噪声" —— 直接用绝对 perf 会把"这条配方拟合出的
        峰值是 98.55 还是 98.91"也算进圈速,与配方偏移(C5 −0.5 / C3 +1.0)重复
        计数,曾导致 **C3 被算成最快配方**;
      * 归一后 perf 只刻画"暖胎 + 衰减"的形状,配方之间的速度差完全由 offset 承担。

    这是 engine.py 早已采用、两个离线脚本却没有跟上的口径(见模块 docstring)。
    """
    if params is None:
        params = FALLBACK_POOLED.get(compound, FALLBACK_POOLED["C4"])
    if base_time is None:
        base_time = BASE_LAP_TIME
    offs = COMPOUND_OFFSET if offsets is None else offsets
    p = perf_pct(tyre_age, params)
    peak = max(float(params.get("peak", 98.5)), 50.0)
    p_norm = max(p / peak, 0.5)
    return (float(base_time) + float(offs.get(compound, 0.0))) / p_norm


def lap_profile(compound, age0, n_laps, base_time, params, offsets=None):
    """一段连续 stint 的逐圈圈速数组(确定性部分,不含随机噪声)。"""
    return np.array([lap_time(compound, age0 + k, base_time, params, offsets)
                     for k in range(int(n_laps))], dtype=np.float64)


# ============================================================================
# 三、DP 求解(离线路径,不含轮胎库存约束)
# ============================================================================
def solve_dp(start_compound, params_by_compound=None, min_stops=1,
             base_time=None, offsets=None, total_laps=None, max_stops=None):
    """动态规划求最优停站策略。

    状态 = (圈数, 配方下标, 胎龄, 已用配方掩码, 停站次数)
    决策 = 继续跑 / 进站换胎(每段 ≥ MIN_STINT_LAPS 圈,末段同样受限)
    目标 = 最小化总比赛时间;约束 = 停站次数 ∈ [min_stops, max_stops] + 至少 2 种配方

    ``max_stops`` 默认 None(不限),此时解出的是**纯时间最优**;
    与 web 的 ``normal``(恰好 1 停)/``two_stop``(恰好 2 停)对比基准对齐时,
    传 ``min_stops=max_stops=1`` 或 ``2``。返回 ``(总秒数, 状态路径)``。
    """
    params_by_compound = params_by_compound or load_pooled_params()
    total = int(TOTAL_LAPS if total_laps is None else total_laps)
    base = BASE_LAP_TIME if base_time is None else float(base_time)
    offsets = COMPOUND_OFFSET if offsets is None else offsets
    n_comp = len(COMPOUNDS)
    inf = float("inf")


    dp = {}
    parent = {}
    start = (0, int(start_compound), 0, 1 << int(start_compound), 0)
    dp[start] = 0.0

    for lap in range(total):
        for state in [s for s in dp if s[0] == lap]:
            _, comp, age, mask, n_stops = state
            t = dp[state]
            cur = lap_time(COMPOUNDS[comp], age, base,
                           params_by_compound[COMPOUNDS[comp]], offsets)

            # 决策一:继续跑(同一套胎,胎龄 +1;超过单套上限则不可继续)
            if age < MAX_TYRE_AGE:
                nxt = (lap + 1, comp, age + 1, mask, n_stops)
                nt = t + cur
                if nxt not in dp or nt < dp[nxt]:
                    dp[nxt] = nt
                    parent[nxt] = state

            # 决策二:进站换胎(受每段最少圈数约束)
            if age < MIN_STINT_LAPS or total - (lap + 1) < MIN_STINT_LAPS:
                continue
            for new_comp in range(n_comp):
                if new_comp == comp:
                    continue
                nxt = (lap + 1, new_comp, 0,
                       mask | (1 << new_comp), n_stops + 1)
                nt = t + cur + PIT_LOSS
                if nxt not in dp or nt < dp[nxt]:
                    dp[nxt] = nt
                    parent[nxt] = state

    best_time, best_state = inf, None
    for state, t in dp.items():
        lap, _comp, _age, mask, n_stops = state
        if lap != total or n_stops < min_stops or bin(mask).count("1") < 2:
            continue
        if max_stops is not None and n_stops > max_stops:
            continue
        if t < best_time:
            best_time, best_state = t, state
    if best_state is None:
        return inf, []

    path = []
    state = best_state
    while state in parent:
        path.append(state)
        state = parent[state]
    path.append(state)
    path.reverse()
    return best_time, path


def solve_dp_best_start(params_by_compound=None, min_stops=1, base_time=None,
                        offsets=None, total_laps=None, max_stops=None):
    """三种起步配方各解一次取最优 → ``(总秒数, 起步配方下标, 状态路径)``。"""
    params_by_compound = params_by_compound or load_pooled_params()
    best = (float("inf"), 0, [])
    for idx in range(len(COMPOUNDS)):
        total, path = solve_dp(idx, params_by_compound, min_stops,
                               base_time, offsets, total_laps, max_stops)
        if path and total < best[0]:
            best = (total, idx, path)
    return best


def extract_stints(path, total_laps=None):
    """从状态路径提取分段: ``[{compound, start_lap, end_lap, laps}, ...]``。

    ``start_lap`` / ``end_lap`` 为**闭区间**(从 0 计),故 ``laps = end - start + 1``。
    旧版 ``plot_cumulative_time`` 用 ``range(start, end)`` 逐圈累加,每段少算 1 圈
    (整条曲线少 停站次数 圈),现统一由 :func:`cumulative_curve` 处理。
    """
    if not path:
        return []
    total = int(TOTAL_LAPS if total_laps is None else total_laps)
    stints = []
    comp = path[0][1]
    start_lap = 0
    for i in range(1, len(path)):
        if path[i][1] != comp:
            stints.append({
                "compound": COMPOUNDS[comp],
                "start_lap": start_lap,
                "end_lap": path[i][0] - 1,
                "laps": path[i][0] - start_lap,
            })
            comp = path[i][1]
            start_lap = path[i][0]
    stints.append({
        "compound": COMPOUNDS[comp],
        "start_lap": start_lap,
        "end_lap": total - 1,
        "laps": total - start_lap,
    })
    return stints


def format_strategy(stints):
    """``[{C5,28}, {C4,50}]`` → ``"C5(28圈) → C4(50圈)"``。"""
    return " → ".join(f"{s['compound']}({s['laps']}圈)" for s in stints)


def cumulative_curve(stints, base_time=None, params_by_compound=None,
                     offsets=None, pit_loss=PIT_LOSS):
    """逐圈累计用时(秒),进站损失计入新 stint 的第一圈。

    返回 ``(圈数序列, 累计用时序列)``,长度 = 总圈数 + 1(第 0 圈 = 0)。
    与 DP 目标函数同源 —— ``solve_dp`` 返回的总秒数应与末值一致,
    这正是 :mod:`tests` 里对账用的不变量(旧图少画了 停站次数 圈)。
    """
    params_by_compound = params_by_compound or load_pooled_params()
    base = BASE_LAP_TIME if base_time is None else float(base_time)
    offsets = COMPOUND_OFFSET if offsets is None else offsets

    xs, ys = [0], [0.0]
    for idx, st in enumerate(stints):
        comp = st["compound"]
        p = params_by_compound.get(comp) or FALLBACK_POOLED.get(comp, FALLBACK_POOLED["C4"])
        for k in range(int(st["laps"])):
            t = lap_time(comp, k, base, p, offsets)
            if idx > 0 and k == 0:
                t += float(pit_loss)
            xs.append(xs[-1] + 1)
            ys.append(ys[-1] + t)
    return xs, ys


if __name__ == "__main__":
    # 冒烟自检: 全场参数 / 车手参数各解一次,并核对 DP 目标值与累计曲线末值一致
    pooled = load_pooled_params()
    drivers = load_driver_params()
    bases = load_base_times()
    print("全场参数:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in pooled.items()})

    total, _idx, path = solve_dp_best_start(pooled, min_stops=1)
    stints = extract_stints(path)
    xs, ys = cumulative_curve(stints)
    ok = abs(ys[-1] - total) < 1e-6
    print(f"纯时间最优(不限停站): {format_strategy(stints)} | DP {total:.1f}s | "
          f"曲线末值 {ys[-1]:.1f}s | 圈数 {xs[-1]} | 对账 {'OK' if ok else '不一致'}")

    t1, _i1, p1 = solve_dp_best_start(pooled, min_stops=1, max_stops=1)
    print(f"1 停基准(对齐 web normal): {format_strategy(extract_stints(p1))} | {t1:.1f}s")

    for code in sorted(bases)[:3]:
        pp = driver_compound_params(drivers, pooled, code)
        t2, _i2, p2 = solve_dp_best_start(pp, min_stops=1, max_stops=1,
                                          base_time=bases[code])
        print(f"  {code}(基准 {bases[code]:.2f}s): {format_strategy(extract_stints(p2))} | {t2:.1f}s")




