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
import sys
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

# ----------------------------------------------------------------- 统一口径
# 轮胎参数 / 圈速公式 / 配方偏移 全部来自 strategy_core(单一事实来源),
# 不再在本文件维护第二份硬编码 PERF_PARAMS。
sys.path.insert(0, SCRIPT_DIR)
import strategy_core as core  # noqa: E402

TOTAL_LAPS = core.TOTAL_LAPS
PIT_LOSS = core.PIT_LOSS                     # 正常进站损失(秒)
PIT_LOSS_SC = core.PIT_LOSS_SC               # SC 下进站损失
PIT_LOSS_VSC = core.PIT_LOSS_VSC             # VSC 下进站损失
COMPOUNDS = core.COMPOUNDS
COMPOUND_OFFSET = core.COMPOUND_OFFSET
N_SIM = 10000                                # 蒙特卡洛模拟次数

# 全场拟合参数(07_拟合参数表.csv,含物理合理性截断)
PERF_PARAMS = core.load_pooled_params()


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
    """性能保持率(%)(实现见 strategy_core)。"""
    return core.perf_pct(t, core.normalize_params(params))


def lap_time(compound, tyre_age, base_time=75.0, params=None):
    """单圈圈速(秒) —— 与 web 端 / DP 同一公式(按 peak 归一 + 配方偏移)。"""
    if params is None:
        params = PERF_PARAMS.get(compound, PERF_PARAMS["C4"])
    return core.lap_time(compound, tyre_age, base_time, params)



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
                      gap_s, remaining, my_base=75.0, rival_base=75.0,
                      sigma=None):
    """你=领导者(先进站)。返回两种分支的胜率。

    胜率 = P(我方最终仍在前),双方各跑 n 圈计入相对波动 sqrt(2n)·σ ——
    只抖进站时间会让结果退化为 0%/100%(确定性赛速),与蒙特卡洛的初衷不符。
    """
    sigma = noise_sigma() if sigma is None else sigma
    m_follow, m_stay = [], []
    for _ in range(N_SIM):
        # A:对手跟进(双方都进站)
        t_me = sample_pit_time()
        t_rival = sample_pit_time() + gap_s
        a_me, a_riv = 0, 0
        for _ in range(remaining):
            t_me += lap_time(my_comp, a_me, my_base)
            t_rival += lap_time(rival_comp, a_riv, rival_base)
            a_me += 1; a_riv += 1
        m_follow.append(t_rival - t_me)

        # B:对手不跟(只有你进站)
        t_me = sample_pit_time()
        t_rival = gap_s
        a_me, a_riv = 0, rival_age
        for _ in range(remaining):
            t_me += lap_time(my_comp, a_me, my_base)
            t_rival += lap_time(rival_comp, a_riv, rival_base)
            a_me += 1; a_riv += 1
        m_stay.append(t_rival - t_me)

    return {"success_follow": _win_rate(m_follow, remaining, sigma),
            "success_stay": _win_rate(m_stay, remaining, sigma)}


# ----------------------------------------------------------------- 场景二:Overcut
def simulate_overcut(my_comp, my_age, rival_comp, rival_age,
                     gap_s, remaining, my_base=75.0, rival_base=75.0,
                     sigma=None):
    """对手=领导者(先进站)。返回两种分支的胜率。"""
    sigma = noise_sigma() if sigma is None else sigma
    m_stay, m_follow = [], []
    for _ in range(N_SIM):
        # A:你留在赛道
        t_me = 0.0; t_rival = sample_pit_time() + gap_s
        a_me, a_riv = my_age, 0
        for _ in range(remaining):
            t_me += lap_time(my_comp, a_me, my_base)
            t_rival += lap_time(rival_comp, a_riv, rival_base)
            a_me += 1; a_riv += 1
        m_stay.append(t_rival - t_me)

        # B:你跟进进站
        t_me = sample_pit_time(); t_rival = sample_pit_time() + gap_s
        a_me, a_riv = 0, 0
        for _ in range(remaining):
            t_me += lap_time(my_comp, a_me, my_base)
            t_rival += lap_time(rival_comp, a_riv, rival_base)
            a_me += 1; a_riv += 1
        m_follow.append(t_rival - t_me)

    return {"success_stay": _win_rate(m_stay, remaining, sigma),
            "success_follow": _win_rate(m_follow, remaining, sigma)}


# ----------------------------------------------------------------- 场景三:SC/VSC
def simulate_sc(my_comp, my_age, rival_comp, rival_age,
                gap_s, remaining, sc_type="SC",
                my_base=75.0, rival_base=75.0, sigma=None):
    """SC/VSC 进站决策。sc_type: "SC" 或 "VSC"。

    安全车下对手几乎必然跟进(损失同样廉价),故对手在这里按"既不进站"的
    双分支比较;收益矩阵里的博弈均衡另见 19/20 两图。
    """
    sigma = noise_sigma() if sigma is None else sigma
    pit_transit = PIT_LANE_TRANSIT_SC if sc_type == "SC" else PIT_LANE_TRANSIT_VSC
    m_pit, m_stay = [], []
    for _ in range(N_SIM):
        # A:你进站(廉价)
        t_me = sample_pit_time(pit_transit); t_rival = gap_s
        a_me, a_riv = 0, rival_age
        for _ in range(remaining):
            t_me += lap_time(my_comp, a_me, my_base)
            t_rival += lap_time(rival_comp, a_riv, rival_base)
            a_me += 1; a_riv += 1
        m_pit.append(t_rival - t_me)

        # B:你不进站
        t_me = 0.0; t_rival = gap_s
        a_me, a_riv = my_age, rival_age
        for _ in range(remaining):
            t_me += lap_time(my_comp, a_me, my_base)
            t_rival += lap_time(rival_comp, a_riv, rival_base)
            a_me += 1; a_riv += 1
        m_stay.append(t_rival - t_me)

    return {"success_pit": _win_rate(m_pit, remaining, sigma),
            "success_stay": _win_rate(m_stay, remaining, sigma)}


def _win_rate(margins, remaining, sigma):
    """P(边际 > 0),边际上叠加双方各跑 n 圈的相对波动 sqrt(2n)·σ。"""
    m = np.asarray(margins, dtype=float)
    noise = np.random.normal(0.0, sigma * np.sqrt(2.0 * max(1, int(remaining))), m.size)
    return float(((m + noise) > 0).mean() * 100.0)


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
def critical_curve(my_comp, my_age, rival_comp, rival_age, gap_s, remaining,
                   my_base=75.0, rival_base=75.0, sigma=None,
                   pit_grid=np.arange(15.0, 35.5, 0.5), n_sim=1000):
    """换胎时间 → Undercut 胜率曲线(与 12 图、临界点同一份计算结果)。

    旧版 ``plot_critical_point`` 另写了一套"C4 vs C3 固定 30 圈"的模拟画图,
    与 ``find_critical_pit_time`` 用的场景不一致 —— 图上曲线与打印的临界点
    其实来自两次不同的计算。现统一由本函数产出。
    """
    sigma = noise_sigma() if sigma is None else sigma
    rates = []
    for pit_s in pit_grid:
        margins = []
        for _ in range(n_sim):
            t_me = float(pit_s)
            t_rival = float(gap_s)
            a_me, a_riv = 0, rival_age
            for _ in range(remaining):
                t_me += lap_time(my_comp, a_me, my_base)
                t_rival += lap_time(rival_comp, a_riv, rival_base)
                a_me += 1; a_riv += 1
            margins.append(t_rival - t_me)
        rates.append(_win_rate(margins, remaining, sigma))
    return pit_grid, rates


def find_critical_pit_time(my_comp, my_age, rival_comp, rival_age,
                           gap_s, remaining, my_base=75.0, rival_base=75.0,
                           threshold=50.0, sigma=None):
    """Undercut 胜率降到 threshold% 以下时的换胎时间(秒)。"""
    grid, rates = critical_curve(my_comp, my_age, rival_comp, rival_age,
                                 gap_s, remaining, my_base, rival_base,
                                 sigma=sigma)
    for pt, r in zip(grid, rates):
        if r < threshold:
            return float(pt)
    return None


# ============================================================================
# 四、收益矩阵与均衡(Stackelberg 的实质实现)
# ============================================================================
# 旧版的做法是"对 2~3 个预设动作各跑一次 MC 比较成功率",既没有最佳反应函数,
# 也没有逆向归纳 —— 严格说称不上"Stackelberg 均衡",只是场景对比。
# 这里显式构造 2×2 收益矩阵并求解:
#
#   我方动作 a ∈ {pit(本圈进站), stay(继续跑)}
#   对手动作 b ∈ {follow(下一圈跟进进站), stay(不进)}
#   收益     = P(我最终仍在前)      —— 头对头位置博弈,零和(对手收益 = 1 − 本方)
#
#   BR_me(b)    = argmax_a M[a][b]                 我方最佳反应
#   BR_rival(a) = argmin_b M[a][b]                 对手最佳反应(最小化我胜率)
#   Nash        = 双方互为最佳反应的格子
#   Stackelberg = 我方先动并承诺 a → 对手取 BR_rival(a) → argmax_a M[a][BR_rival(a)]
#
# 关键区别:胜率不再由"两张固定的成功率数字"给出,而是由收益矩阵 + 反应函数推出,
# 因而可以画出"哪些 (差距, 胎龄) 区域里先进站是占优/均衡策略"(20 图)。

ME_ACTIONS = ("pit", "stay")
RIVAL_ACTIONS = ("follow", "stay")


def noise_sigma(default=0.7357):
    """单圈随机波动 σ(秒): 优先读 14_圈速波动估计.csv 的汇总(1.5IQR)行。"""
    path = os.path.join(CHART_DIR, "14_圈速波动估计.csv")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        hit = df[df["范围"].astype(str).str.startswith("汇总(1.5IQR")]
        if len(hit):
            return float(hit["sigma_s"].iloc[0])
    except (OSError, KeyError, ValueError):
        pass
    return float(default)


def _pit_loss_samples(rng, n):
    """进站总损失抽样(秒) = 通道行驶 + 换胎混合分布。"""
    normal = rng.random(n) < PIT_NORMAL_PROB
    change = np.where(
        normal,
        np.maximum(1.5, rng.normal(PIT_NORMAL_MEAN, PIT_NORMAL_STD, n)),
        np.maximum(3.0, rng.normal(PIT_ABNORMAL_MEAN, PIT_ABNORMAL_STD, n)),
    )
    return PIT_LANE_TRANSIT + change


def _cum_cost(compound, age0, n_laps, base_time):
    """从胎龄 age0 起连续 n_laps 圈的累计耗时,长度 n_laps+1。"""
    tt = np.array([lap_time(compound, age0 + k, base_time)
                   for k in range(int(n_laps))], dtype=float)
    return np.concatenate([[0.0], np.cumsum(tt)])


# 摩纳哥超车难度(与 web/engine.py 的 monaco_pass_* 同一标定):
# 3 年摩纳哥正赛赛道超车仅个位数 → 单圈超车概率 ≈ 0.5% + 0.8%/每 1s 速度优势,
# 上限 6%。这决定了"进站让出位置"在摩纳哥的代价。
MONACO_PASS_BASE = 0.005
MONACO_PASS_SLOPE = 0.008
MONACO_PASS_MAX = 0.06


def pass_probability(pace_gain_s_per_lap, remaining, base=None, slope=None, cap=None):
    """在剩余圈数内完成一次超车的概率(速度优势 → 每圈超车概率 → 累积)。"""
    base = MONACO_PASS_BASE if base is None else base
    slope = MONACO_PASS_SLOPE if slope is None else slope
    cap = MONACO_PASS_MAX if cap is None else cap
    p_lap = min(max(base + slope * max(float(pace_gain_s_per_lap), 0.0), base), cap)
    return float(1.0 - (1.0 - p_lap) ** max(int(remaining), 1))


def payoff_matrix(my_comp, my_age, rival_comp, rival_age, gap_s, remaining,
                  my_base=75.0, rival_base=75.0, n_sim=N_SIM, sigma=None, seed=7):
    """2×2 收益矩阵: ``M[(我方动作, 对手动作)] = P(我最终位次仍在前)``。

    模型由两层组成 —— 这是"摩纳哥"这个题目的关键:

    1. **时间层**: 双方剩余圈数的耗时(含轮胎衰减)+ 进站损失 + 当前领先差距,
       得到终点的净时间差 ``margin``;
    2. **位置层**: 进站会让出赛道位置,而摩纳哥几乎无法超车(单圈超车概率
       :data:`MONACO_PASS_BASE` 起、随速度优势线性上升、上限 6%)。
       因此"只有我进站"时,必须在剩余圈内**追近并完成超车**才能赢:
       ``P(赢) = P(margin>0) × P(超车)``;反之"只有对手进站"时他必须超我。

    若忽略第 2 层(纯时间模型),会得出"胎龄 30 圈时先进站胜率 91%"这种与摩纳哥
    现实完全相反的结论 —— 停站在这里首先损失的是位置,其次才是时间。

    * 我方 ``pit`` → 本圈进站;``stay`` → 用旧胎跑完
    * 对手 ``follow`` → **下一圈**才进站(跟进存在一圈反应延迟)
    * ``gap_s > 0`` 表示我当前领先该秒数
    * ``sigma`` → 单圈随机波动,双方各跑 n 圈的相对波动 ≈ sqrt(2n)·σ
    """
    rng = np.random.default_rng(seed)
    R = int(max(1, remaining))
    if sigma is None:
        sigma = noise_sigma()

    me_pit_cum = _cum_cost(my_comp, 0, R, my_base)
    me_stay_cum = _cum_cost(my_comp, my_age, R, my_base)
    riv_stay_cum = _cum_cost(rival_comp, rival_age, R, rival_base)
    if R >= 2:
        riv_follow_cum = (_cum_cost(rival_comp, rival_age, 1, rival_base)[-1]
                          + _cum_cost(rival_comp, 0, R - 1, rival_base)[-1])
    else:
        riv_follow_cum = riv_stay_cum[-1]

    total_me = {"stay": me_stay_cum[-1], "pit": me_pit_cum[-1]}
    total_riv = {"stay": riv_stay_cum[-1], "follow": riv_follow_cum}

    # 各分支下的平均圈速差(用于算超车概率),正 = 我方更快
    pace_me_pit = (total_riv["stay"] - total_me["pit"]) / R
    pace_riv_follow = (total_me["stay"] - total_riv["follow"]) / R

    noise = rng.normal(0.0, sigma * np.sqrt(2.0 * R), n_sim)
    matrix = {}
    for a in ME_ACTIONS:
        my_loss = _pit_loss_samples(rng, n_sim) if a == "pit" else 0.0
        for b in RIVAL_ACTIONS:
            riv_loss = _pit_loss_samples(rng, n_sim) if b == "follow" else 0.0
            margin = (total_riv[b] + riv_loss) - (total_me[a] + my_loss) + gap_s + noise
            p_ahead = float((margin > 0).mean())

            if a == "pit" and b == "stay":
                # 只有我进站: 位置让出,必须追近并完成超车
                p = p_ahead * pass_probability(pace_me_pit, R)
            elif a == "stay" and b == "follow":
                # 只有对手进站: 他必须追近并超我
                p = 1.0 - (1.0 - p_ahead) * pass_probability(pace_riv_follow, R)
            else:
                # 双方同向(都进站/都不进站): 位置由停站时序或初始位置决定
                p = p_ahead
            matrix[(a, b)] = float(p)
    return matrix



def solve_game(matrix):
    """由收益矩阵求最佳反应、Nash 均衡与 Stackelberg 结果(我方为领导者)。"""
    br_me = {b: max(ME_ACTIONS, key=lambda a: matrix[(a, b)]) for b in RIVAL_ACTIONS}
    br_rival = {a: min(RIVAL_ACTIONS, key=lambda b: matrix[(a, b)]) for a in ME_ACTIONS}
    nash = [(a, b) for a in ME_ACTIONS for b in RIVAL_ACTIONS
            if br_me[b] == a and br_rival[a] == b]
    leader = max(ME_ACTIONS, key=lambda a: matrix[(a, br_rival[a])])
    follower = br_rival[leader]
    return {
        "br_me": br_me,
        "br_rival": br_rival,
        "nash": nash,
        "leader_action": leader,
        "follower_action": follower,
        "stackelberg_value": matrix[(leader, follower)],
        "commit_best": matrix[(leader, follower)],
        "simultaneous_guess": matrix[(br_me[follower], follower)],
    }


ACTION_ZH = {"pit": "先进站(undercut)", "stay": "继续跑(保位置)",
             "follow": "跟进进站", "stay_rival": "不进站"}


def equilibrium_region(gaps, ages, my_comp="C4", rival_comp="C4", rival_age=12,
                       current_lap=45, my_base=75.0, rival_base=75.0,
                       n_sim=4000, sigma=None):
    """在 (当前差距, 我方胎龄) 网格上逐点求均衡,返回明细行。

    每行记录: 我方 Stackelberg 最优动作、对手最优反应、该格胜率,以及四个格子的
    胜率 —— 便于画"均衡区域图"(哪些区域里先进站是均衡策略)。
    """
    sigma = noise_sigma() if sigma is None else sigma
    remaining = TOTAL_LAPS - current_lap
    rows = []
    for gap in gaps:
        for age in ages:
            m = payoff_matrix(my_comp, age, rival_comp, rival_age, gap, remaining,
                              my_base, rival_base, n_sim=n_sim, sigma=sigma)
            g = solve_game(m)
            rows.append({
                "当前圈": current_lap,
                "我方胎龄": age,
                "差距_s": round(float(gap), 2),
                "剩余圈数": remaining,
                "均衡动作": ACTION_ZH.get(g["leader_action"], g["leader_action"]),
                "对手最佳反应": ACTION_ZH.get(g["follower_action"],
                                              g["follower_action"]),
                "均衡胜率_%": round(g["stackelberg_value"] * 100, 1),
                "进站_对手跟进_%": round(m[("pit", "follow")] * 100, 1),
                "进站_对手不进_%": round(m[("pit", "stay")] * 100, 1),
                "继续_对手跟进_%": round(m[("stay", "follow")] * 100, 1),
                "继续_对手不进_%": round(m[("stay", "stay")] * 100, 1),
                "Nash均衡": "/".join(f"{a}-{b}" for a, b in g["nash"]) or "无纯策略",
            })
    return rows


# ----------------------------------------------------------------- 可视化
def plot_payoff_matrix(matrix, out_path, gap_s=None, remaining=None):
    """2×2 收益矩阵热力图(行=我方动作,列=对手动作)。"""
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    grid = np.array([[matrix[("pit", "follow")], matrix[("pit", "stay")]],
                     [matrix[("stay", "follow")], matrix[("stay", "stay")]]])
    im = ax.imshow(grid * 100, cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks([0, 1], ["对手跟进进站", "对手不进站"])
    ax.set_yticks([0, 1], ["我方先进站", "我方继续跑"])
    for i in range(2):
        for j in range(2):
            v = grid[i, j] * 100
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="black" if 25 < v < 85 else "white")
    # 标注最佳反应: 每列最优行(我方)
    for j in range(2):
        i = int(np.argmax(grid[:, j]))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#1f6fb2", lw=2.5))
    title = "2×2 收益矩阵: 我方最终仍领先的概率"
    if gap_s is not None:
        title += f"\n(当前领先 {gap_s:+.1f}s,剩余 {remaining} 圈)"
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, label="我方位次保持概率 (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


def plot_equilibrium_region(rows, out_path):
    """均衡区域图: x=当前差距, y=我方胎龄, 颜色=均衡是否先进站。"""
    gaps = sorted({r["差距_s"] for r in rows})
    ages = sorted({r["我方胎龄"] for r in rows})
    pit_grid = np.zeros((len(ages), len(gaps)))
    val_grid = np.zeros((len(ages), len(gaps)))
    for r in rows:
        i = ages.index(r["我方胎龄"])
        j = gaps.index(r["差距_s"])
        pit_grid[i, j] = 1.0 if r["均衡动作"].startswith("先进站") else 0.0
        val_grid[i, j] = r["均衡胜率_%"] / 100.0

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.imshow(pit_grid, origin="lower", aspect="auto", cmap="coolwarm",
              vmin=0, vmax=1,
              extent=[gaps[0], gaps[-1], ages[0], ages[-1]])
    cs = ax.contour(gaps, ages, val_grid, levels=[0.5, 0.7, 0.9],
                    colors="black", linewidths=0.8, linestyles=":")
    ax.clabel(cs, fmt="%.1f", fontsize=8)
    for r in rows:
        ax.text(r["差距_s"], r["我方胎龄"], f"{r['均衡胜率_%']:.0f}",
                ha="center", va="center", fontsize=7,
                color="white" if r["均衡胜率_%"] < 50 else "black")
    ax.set_xlabel("当前与对手差距 (秒,正=我方领先)")
    ax.set_ylabel("我方轮胎胎龄 (圈)")
    ax.set_title("均衡区域: 红色=先进站(undercut)为均衡, 蓝色=继续跑为均衡\n"
                 "数字为 Stackelberg 均衡胜率(%)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


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


def plot_critical_point(critical_pit, out_path, grid=None, rates=None):
    """关键拐点标注图(曲线与临界点来自同一次计算 ``critical_curve``)。"""
    if grid is None or rates is None:
        grid, rates = critical_curve("C4", 30, "C4", 30, 0.8, TOTAL_LAPS - 45,
                                     74.7, 74.7)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grid, rates, lw=2, color="#1f6fb2")
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
        "my_comp": "C4", "my_age": 30,
        "rival_comp": "C4", "rival_age": 30,
        "gap_s": 0.8,
        "remaining": TOTAL_LAPS - 45,
        "my_base": 74.7, "rival_base": 74.7,
    }
    print(f"\n  当前状态(示例: 第45圈,1v1 同配方同胎龄的\"谁先进站\"局面)")
    print(f"  我:   {params['my_comp']} (胎龄{params['my_age']}圈)")
    print(f"  对手: {params['rival_comp']} (胎龄{params['rival_age']}圈)")
    print(f"  差距: {params['gap_s']}秒(正=我方领先) | 剩余: {params['remaining']}圈")

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
    grid, rates = critical_curve(**params)
    crit = next((float(p) for p, r in zip(grid, rates) if r < 50.0), None)
    if crit:
        print(f"  换胎时间 > {crit:.1f}秒 → Undercut 胜率降至 50% 以下")
    else:
        print("  未找到临界点(全区间内胜率均 ≥50%)")

    print("\n[输出]")
    all_results = {"undercut": undercut, "overcut": overcut, "sc": sc}
    plot_decision_comparison(all_results,
                             os.path.join(CHART_DIR, "11_Stackelberg决策结果.png"))
    plot_tire_choice(tire_times, params["remaining"],
                     os.path.join(CHART_DIR, "13_轮胎选择.png"))
    plot_critical_point(crit, os.path.join(CHART_DIR, "12_关键拐点.png"),
                        grid=grid, rates=rates)

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

    # ---------------------------------------------------------------- 收益矩阵与均衡
    print("\n[收益矩阵与均衡(2×2 + 逆向归纳)]")
    sigma = noise_sigma()
    m = payoff_matrix(**params, sigma=sigma)
    game = solve_game(m)
    print(f"  σ(单圈波动) = {sigma:.4f}s")
    for a in ME_ACTIONS:
        for b in RIVAL_ACTIONS:
            print(f"  我方[{ACTION_ZH[a]}] × 对手[{ACTION_ZH[b]}] → 胜率 {m[(a, b)]*100:.1f}%")
    print(f"  我方最佳反应: 对手跟进→{ACTION_ZH[game['br_me']['follow']]} | "
          f"对手不进→{ACTION_ZH[game['br_me']['stay']]}")
    print(f"  对手最佳反应: 我方先进站→{ACTION_ZH[game['br_rival']['pit']]} | "
          f"我方继续跑→{ACTION_ZH[game['br_rival']['stay']]}")
    nash_txt = ("/".join(f"{ACTION_ZH[a]}+{ACTION_ZH[b]}" for a, b in game["nash"])
                or "无纯策略均衡(混合策略)")
    print(f"  Nash 均衡: {nash_txt}")
    print(f"  Stackelberg 解(我方先动): {ACTION_ZH[game['leader_action']]} → "
          f"对手{ACTION_ZH[game['follower_action']]}, 均衡胜率 "
          f"{game['stackelberg_value']*100:.1f}%")

    payoff_path = os.path.join(CHART_DIR, "19_收益矩阵.csv")
    pd.DataFrame([
        {"我方动作": ACTION_ZH[a], "对手动作": ACTION_ZH[b],
         "我方胜率_%": round(m[(a, b)] * 100, 1),
         "对手胜率_%": round((1 - m[(a, b)]) * 100, 1)}
        for a in ME_ACTIONS for b in RIVAL_ACTIONS
    ]).to_csv(payoff_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {os.path.basename(payoff_path)}")
    plot_payoff_matrix(m, os.path.join(CHART_DIR, "19_收益矩阵.png"),
                       gap_s=params["gap_s"], remaining=params["remaining"])

    # 均衡区域:(差距 −3~+6 秒) × (我方胎龄 5~45 圈)
    gaps = np.arange(-3.0, 6.5, 1.0)
    ages = list(range(5, 46, 5))
    region = equilibrium_region(gaps, ages,
                               my_comp=params["my_comp"],
                               rival_comp=params["rival_comp"],
                               rival_age=params["rival_age"],
                               current_lap=45,
                               my_base=params["my_base"],
                               rival_base=params["rival_base"],
                               sigma=sigma)
    region_path = os.path.join(CHART_DIR, "20_均衡区域.csv")
    pd.DataFrame(region).to_csv(region_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {os.path.basename(region_path)}")
    plot_equilibrium_region(region, os.path.join(CHART_DIR, "20_均衡区域图.png"))

    pit_cells = sum(1 for r in region if r["均衡动作"].startswith("先进站"))
    print(f"  均衡区域: {pit_cells}/{len(region)} 个 (差距,胎龄) 组合上"
          f"\"先进站\"是均衡策略")

    print("\n完成!")


if __name__ == "__main__":
    main()