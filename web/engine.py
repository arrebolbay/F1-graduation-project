#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
F1 摩纳哥站进站策略优化系统 — Web 后端核心引擎
=================================================
为前端提供两类能力:
  1. 赛前模拟(问题一): 读取 DP 求解结果(10b/10c/10d),解析分段策略,
     计算累计用时曲线与对比指标
  2. 赛中决策(问题二): 参数化 Stackelberg 蒙特卡洛
     (Undercut / Overcut / SC·VSC / 红旗),逐圈追踪"完成超越圈数"与被反超风险

数据来源(全部为项目内已生成的 CSV):
  图表\07_拟合参数表.csv          全场汇总轮胎衰减参数
  图表\07b_车手衰减参数表.csv     分车手分配方衰减参数
  图表\07c_车手驾驶习惯指标.csv   车手基准圈速/风格
  图表\10b_车手策略对比表.csv     车手最优(正常)策略
  图表\10c_车手强制两停对比表.csv 车手强制两停策略
  图表\10d_正常vs两停对比表.csv   正常 vs 两停对比

模型与 脚本\solve_stackelberg.py 保持一致(性能保持率衰减模型、
进站混合分布抽样、分场景分支语义),扩展点:
  - 车手级参数(基准圈速 + 分配方衰减)替代全场统一参数
  - 可调参数(进站通道/换胎分布/配方偏移/圈速波动)
  - 超越圈数追踪: 反超后不再被反超的"永久反超圈"
"""

import csv
import json
import os
import threading
import time

import numpy as np

# ----------------------------------------------------------------- 路径
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(WEB_DIR)
CHART_DIR = os.path.join(PROJECT_DIR, "图表")
HISTORY_FILE = os.path.join(WEB_DIR, "history.jsonl")

# ----------------------------------------------------------------- 全局参数
TOTAL_LAPS = 78
COMPOUNDS = ["C5", "C4", "C3"]
COMPOUND_COLORS = {"C5": "#e10600", "C4": "#ffd24d", "C3": "#b8bdc7"}
COMPOUND_ZH = {"C5": "C5 软胎", "C4": "C4 中性胎", "C3": "C3 硬胎"}

# 蒙特卡洛默认参数(可在前端"高级参数"覆盖)
# lap_noise_std 默认 1.05 = FastF1 2022-2025 摩纳哥正赛 stint 去趋势残差
# (1.5IQR 去尾后)的 σ 实测值,详见 图表\14_圈速波动估计.csv —— 近似正态。
DEFAULT_PARAMS = {
    "pit_lane_transit": 16.5,        # 正常进站通道行驶(秒)
    "pit_lane_transit_scvsc": 6.0,   # SC/VSC 窗口通道(秒,两者损失接近,合并建模)
    "pit_lane_transit_red": 0.0,     # 红旗免费换胎(比赛暂停,无时间损失)
    "pit_normal_prob": 0.85,         # 正常换胎概率
    "pit_normal_mean": 2.5,          # 正常换胎耗时均值(秒)
    "pit_normal_std": 0.3,           # 正常换胎耗时标准差(秒)
    "pit_abnormal_mean": 5.0,        # 异常换胎耗时均值(秒)
    "pit_abnormal_std": 2.0,         # 异常换胎耗时标准差(秒)
    "compound_offset": {"C3": 1.0, "C4": 0.0, "C5": -0.5},  # 配方速度偏移(秒)
    "lap_noise_std": 1.05,           # 单圈随机波动 σ(秒),数据标定值
    "form_noise_std": 0.20,          # "当日状态"相关波动 σ(秒/圈,整段恒定,防胜率饱和)
    "red_restart_gain": 0.35,        # 红旗静态发车新胎抓地增益(秒/圈)
    "red_restart_laps": 2,           # 发车增益持续圈数
    "pit_reaction_laps": 2,          # 跟进进站反应延迟(圈,Undercut/Overcut 机制来源)
    "n_sim": 10000,                  # 蒙特卡洛次数
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

SCENARIOS = {
    "undercut": {
        "name": "Undercut",
        "title": "Undercut · 你先进站",
        "desc": "你=博弈领导者(先进站),对手=跟随者(决定跟不跟)",
    },
    "overcut": {
        "name": "Overcut",
        "title": "Overcut · 对手先进站",
        "desc": "对手=博弈领导者(先进站),你=跟随者(留在赛道或跟进)",
    },
    "sc": {
        "name": "SC/VSC",
        "title": "SC / VSC · 安全车窗口",
        "desc": "安全车/虚拟安全车出动(两者进站损失几乎相同,合并建模),决策是否利用廉价进站窗口",
    },
    "red": {
        "name": "红旗",
        "title": "红旗 · 免费换胎 + 静态发车",
        "desc": "比赛暂停,可免费换胎且恢复后静态发车;换上新/软胎可获发车抓地优势",
    },
}

_HISTORY_LOCK = threading.Lock()

# ----------------------------------------------------------------- 数据加载
def _read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class DataStore:
    """一次性加载全部 CSV,提供车手/策略查询。"""

    def __init__(self):
        self.pooled = self._load_pooled()
        self.deg = self._load_deg()
        self.metrics = self._load_metrics()
        self.strategies = self._load_strategies()      # {driver: {normal/two_stop}}
        self.compare = self._load_compare()            # {driver: {...}}

    # ---- 07_拟合参数表.csv(全场汇总,缺数据车手回退)
    def _load_pooled(self):
        pooled = {}
        path = os.path.join(CHART_DIR, "07_拟合参数表.csv")
        for row in _read_csv(path):
            comp = (row.get("配方") or "").strip()
            if comp in COMPOUNDS:
                pooled[comp] = {
                    "P0": _f(row.get("P0_pct")),
                    "t_peak": _f(row.get("t_peak_laps")),
                    "peak": _f(row.get("peak_pct")),
                    "deg": _f(row.get("deg_rate_pct_per_lap")),
                }
        return pooled

    # ---- 07b_车手衰减参数表.csv
    def _load_deg(self):
        deg = {}
        path = os.path.join(CHART_DIR, "07b_车手衰减参数表.csv")
        for row in _read_csv(path):
            code = (row.get("车手") or "").strip().upper()
            comp = (row.get("配方") or "").strip()
            if code and comp in COMPOUNDS:
                deg.setdefault(code, {})[comp] = {
                    "P0": _f(row.get("P0_pct")),
                    "t_peak": _f(row.get("t_peak_laps")),
                    "peak": _f(row.get("peak_pct")),
                    "deg": _f(row.get("deg_rate_pct_per_lap")),
                    "source": (row.get("来源") or "").strip(),
                }
        return deg

    # ---- 07c_车手驾驶习惯指标.csv
    def _load_metrics(self):
        metrics = {}
        path = os.path.join(CHART_DIR, "07c_车手驾驶习惯指标.csv")
        for row in _read_csv(path):
            code = (row.get("车手") or "").strip().upper()
            if not code:
                continue
            metrics[code] = {
                "code": code,
                "name": (row.get("中文名") or DRIVER_NAMES.get(code, code)).strip(),
                "base_time": _f(row.get("基准圈速_s"), 75.5),
                "style": (row.get("驾驶风格") or "均衡型").strip(),
                "conserve": _f(row.get("保胎评分")),
                "deg_rate": _f(row.get("平均衰减率_pct_per_lap")),
            }
        return metrics

    # ---- 策略字符串解析: "C4(73圈) → C3(5圈)"
    @staticmethod
    def parse_strategy_text(text):
        stints = []
        start = 1
        for part in (text or "").split("→"):
            part = part.strip()
            comp, laps = None, 0
            if "(" in part and "圈" in part:
                comp = part.split("(")[0].strip().upper()
                num = part.split("(")[1].split("圈")[0].strip()
                try:
                    laps = int(num)
                except ValueError:
                    laps = 0
            if comp in COMPOUNDS and laps > 0:
                stints.append({"compound": comp, "laps": laps,
                               "start": start, "end": start + laps - 1})
                start += laps
        return stints

    # ---- 10b / 10c
    def _load_strategies(self):
        strategies = {}
        for mode, fname in (("normal", "10b_车手策略对比表.csv"),
                            ("two_stop", "10c_车手强制两停对比表.csv")):
            path = os.path.join(CHART_DIR, fname)
            for row in _read_csv(path):
                code = (row.get("车手") or "").strip().upper()
                if not code:
                    continue
                text = (row.get("最优策略") or "").strip()
                strategies.setdefault(code, {})[mode] = {
                    "text": text,
                    "stops": int(_f(row.get("停站次数"))),
                    "total_s": _f(row.get("总秒数")),
                    "total_display": (row.get("总用时") or "").strip(),
                    "stints": self.parse_strategy_text(text),
                }
        return strategies

    # ---- 10d_正常vs两停对比表.csv
    def _load_compare(self):
        compare = {}
        path = os.path.join(CHART_DIR, "10d_正常vs两停对比表.csv")
        for row in _read_csv(path):
            code = (row.get("车手") or "").strip().upper()
            if not code:
                continue
            delta = (row.get("两停损失_s") or "0").strip().replace("+", "")
            compare[code] = {
                "normal_text": (row.get("正常策略") or "").strip(),
                "normal_s": _f(row.get("正常用时_s")),
                "two_text": (row.get("两停策略") or "").strip(),
                "two_s": _f(row.get("两停用时_s")),
                "delta_s": _f(delta),
            }
        return compare

    # ---- 查询接口
    def driver_list(self):
        """供下拉框使用的车手列表(按基准圈速升序)。"""
        codes = sorted(set(self.metrics) | set(self.strategies))
        out = []
        for code in codes:
            m = self.metrics.get(code, {})
            out.append({
                "code": code,
                "name": m.get("name", DRIVER_NAMES.get(code, code)),
                "style": m.get("style", "—"),
                "base_time": m.get("base_time", 75.5),
                "has_strategy": code in self.strategies,
                "has_metrics": code in self.metrics,
            })
        out.sort(key=lambda d: d["base_time"])
        return out

    def deg_params(self, code, compound):
        """车手级衰减参数,缺则回退全场汇总(含暖胎段物理合理性截断)。"""
        raw = self.deg.get(code, {}).get(compound) or self.pooled.get(
            compound, self.pooled.get("C4", {"P0": 98.0, "t_peak": 0.0,
                                             "peak": 98.5, "deg": 0.03}))
        return _sanitize_deg(raw)

    def base_time(self, code):
        return self.metrics.get(code, {}).get("base_time", 75.5)


STORE = DataStore()


def _sanitize_deg(params):
    """暖胎段物理合理性截断。

    拟合的 P0→peak 爬升段混入了发车/交通等混杂效应(个别拟合出现
    t_peak 高达 22 圈、新胎比峰值慢 3~4 个点 → 新硬胎首圈竟慢 3 秒、
    轮胎"越跑越快"20 余圈的非物理结果)。真实暖胎约为 1~3 圈、
    落差 ≤1.2 个性能点,故引擎端统一截断:
      t_peak ≤ 3 圈; P0 ≥ peak − 1.2。
    衰减段斜率 deg 保持拟合值不变。
    """
    p = dict(params)
    peak = float(p.get("peak", 98.5))
    p["t_peak"] = min(max(float(p.get("t_peak", 0.0)), 0.0), 3.0)
    p["P0"] = max(float(p.get("P0", peak)), peak - 1.2)
    return p

# ----------------------------------------------------------------- 圈速波动(实测标定)
# 图表\14_圈速波动估计.csv(脚本\fit_lap_noise.py 产出):
#   FastF1 2022-2025 摩纳哥正赛 stint 去趋势残差,1.5IQR 去尾后
#   σ = 1.05s 且近似正态(正态检验 p=0.09) —— 成功率由此为连续概率而非硬判定。
#   分车手 σ 为原始值,按"车手/全场 原始σ 比值"缩放到去尾基准 → 车手级加权 σ。
_NOISE_TRIM = 1.0547          # 全场去尾 σ(正态基准)
_NOISE_RAW_ALL = 2.8833       # 全场原始 σ(缩放分母)
_NOISE_BY_DRIVER = {}         # 车手 -> 原始 σ


def _load_noise_table():
    global _NOISE_TRIM, _NOISE_RAW_ALL
    path = os.path.join(CHART_DIR, "14_圈速波动估计.csv")
    if not os.path.exists(path):
        return
    for row in _read_csv(path):
        scope = (row.get("范围") or "").strip()
        key = (row.get("键") or "").strip().upper()
        try:
            s = float(row.get("sigma_s"))
        except (TypeError, ValueError):
            continue
        if scope.startswith("汇总(1.5IQR"):
            _NOISE_TRIM = s
        elif scope.startswith("汇总(原始"):
            _NOISE_RAW_ALL = s
        elif scope == "车手":
            _NOISE_BY_DRIVER[key] = s


_load_noise_table()


def driver_noise(code):
    """车手级单圈波动 σ(秒): 实测分车手 σ 相对比值 × 去尾正态基准,限幅 [0.4, 2.5]。"""
    raw = _NOISE_BY_DRIVER.get((code or "").strip().upper())
    if not raw or raw <= 0 or _NOISE_RAW_ALL <= 0:
        return round(_NOISE_TRIM, 3)
    return round(min(max(_NOISE_TRIM * raw / _NOISE_RAW_ALL, 0.4), 2.5), 3)

# ----------------------------------------------------------------- 轮胎模型
def perf_pct(t, params):
    """性能保持率(%): 暖胎上升 → 峰值 → 线性衰减(与主脚本一致)。"""
    P0, peak, t_peak, deg = params["P0"], params["peak"], params["t_peak"], params["deg"]
    if t <= t_peak:
        return P0 + (peak - P0) * (t / max(t_peak, 1.0))
    return peak - deg * (t - t_peak)


def lap_time(compound, tyre_age, base_time, params, offsets=None):
    """单圈圈速(秒) = (基准圈速 + 配方偏移) ÷ 归一化性能保持率。

    perf 曲线按各自峰值归一(p/peak,1.0=峰值状态)——消除跨配方的拟合
    水平噪声(直接用绝对 perf 会与配方偏移双重计数,曾导致 C3 被算成最快
    配方);归一后 perf 只刻画"暖胎 + 衰减"形状,配方速度差由 offset 承担。
    """
    offsets = offsets if offsets is not None else DEFAULT_PARAMS["compound_offset"]
    p = perf_pct(tyre_age, params)
    peak = max(float(params.get("peak", 98.5)), 50.0)
    p_norm = max(p / peak, 0.5)
    return (base_time + offsets.get(compound, 0.0)) / p_norm


def lap_profile(compound, age0, n_laps, base_time, params, offsets=None):
    """一段连续 stint 的逐圈圈速数组(确定性部分)。"""
    return np.array([lap_time(compound, age0 + k, base_time, params, offsets)
                     for k in range(n_laps)], dtype=np.float64)


def sample_pit_times(rng, n, transit, p):
    """进站总损失抽样(秒) = 通道行驶 + 换胎混合分布(与主脚本一致)。"""
    normal = rng.random(n) < p["pit_normal_prob"]
    change = np.where(
        normal,
        np.maximum(1.5, rng.normal(p["pit_normal_mean"], p["pit_normal_std"], n)),
        np.maximum(3.0, rng.normal(p["pit_abnormal_mean"], p["pit_abnormal_std"], n)),
    )
    return transit + change


# ----------------------------------------------------------------- 蒙特卡洛
def _run_branch(rng, n_sim, remaining, lt_me, lt_riv, t0_me, t0_riv,
                noise_me, noise_riv, form_me=0.0, form_riv=0.0):
    """
    逐圈推进两车"距完赛时间",圈速随机性分两层:
      - 逐圈独立噪声 N(0, σ_lap²)(FastF1 实测标定);
      - "当日状态"相关项 N(0, σ_form²) —— 每车每次模拟抽取一次、整段恒定
        的圈速偏移,补偿独立噪声对"整段节奏波动"(交通/状态/赛道演变)的低估,
        避免总时间分布过窄导致胜率饱和到 0%/100%。
    记录:
      success     最终我方先完赛
      overtake    我方永久反超圈 = 最后一次落后之后的第1圈(0=自始至终领先)
      overtaken   对手永久反超圈 = 我方最后一次领先之后的第1圈(0=我方未被反超)
      swapped     比赛中双方位置是否至少变化过一次(被反超风险核心指标)
    """
    t_me = np.asarray(t0_me, dtype=np.float64).copy()
    t_riv = np.asarray(t0_riv, dtype=np.float64).copy()
    ahead0 = t_me < t_riv
    last_behind = np.where(~ahead0, 0, -1)   # 我方落后的最后圈号(0=起点)
    last_ahead = np.where(ahead0, 0, -1)     # 我方领先的最后圈号
    swapped = np.zeros(n_sim, dtype=bool)
    order = ahead0.copy()
    form_m = rng.normal(0, form_me, n_sim) if form_me > 0 else 0.0
    form_r = rng.normal(0, form_riv, n_sim) if form_riv > 0 else 0.0
    for k in range(remaining):
        add_me = (lt_me[k] + form_m if noise_me <= 0
                  else lt_me[k] + form_m + rng.normal(0, noise_me, n_sim))
        add_riv = (lt_riv[k] + form_r if noise_riv <= 0
                   else lt_riv[k] + form_r + rng.normal(0, noise_riv, n_sim))
        t_me += add_me
        t_riv += add_riv
        now_ahead = t_me < t_riv
        swapped |= (now_ahead != order)
        order = now_ahead
        last_behind = np.where(~now_ahead, k + 1, last_behind)
        last_ahead = np.where(now_ahead, k + 1, last_ahead)
    success = t_me < t_riv
    overtake = last_behind + 1
    # 被反超圈: 最终失去位置、且中途曾领先 → 反超发生在最后一次领先之后
    overtaken = np.where((~success) & (last_ahead >= 0), last_ahead + 1, 0)
    return success, overtake, overtaken, swapped, t_riv - t_me


def _branch_stats(success, overtake, overtaken, swapped, delta, remaining):
    """单分支统计: 成功率 + 得而复失风险 + 反超圈数分布。

    风险口径: "被反超风险" = 得而复失率 = P(曾领先却最终丢位置 | 曾获得位置优势);
    swap_rate 仅为"位置翻转频繁度"参考(包含我方成功超越造成的翻转,不计入风险)。
    """
    n = len(success)
    n_succ = int(success.sum())
    gained = success | (overtaken > 0)          # 曾获得位置优势的样本
    n_gained = int(gained.sum())
    n_relost = int((overtaken > 0).sum())       # 曾领先但最终丢掉(得而复失)
    relost_rate = (n_relost / n_gained * 100.0) if n_gained else 0.0
    stats = {
        "success_rate": round(n_succ / n * 100.0, 2) if n else 0.0,
        "n_success": n_succ,
        "mean_finish_delta": round(float(delta.mean()), 3),   # 正=我方先完赛
        # ---- 被反超风险(条件概率口径)
        "loss_rate": round(float((~success).mean() * 100.0), 2),
        "gained_rate": round(n_gained / n * 100.0, 2) if n else 0.0,
        "relost_rate": round(relost_rate, 2),                 # 得而复失率(核心风险指标)
        "swap_rate": round(float(swapped.mean() * 100.0), 2),  # 翻转频繁度(仅供参考)
    }
    lost_lead = (~success) & (overtaken > 0)
    n_lost_lead = int(lost_lead.sum())
    stats["overtaken_rate"] = round(n_lost_lead / n * 100.0, 2) if n else 0.0
    if n_lost_lead:
        ot = overtaken[lost_lead].astype(np.int64)
        stats["overtaken_mean"] = round(float(ot.mean()), 2)
        stats["overtaken_median"] = float(np.median(ot))
    else:
        stats["overtaken_mean"] = None
        stats["overtaken_median"] = None
    # 风险等级: 按得而复失率(曾领先后又丢掉的条件概率)评定
    stats["risk_level"] = ("高" if relost_rate >= 30 else
                           "中" if relost_rate >= 10 else "低")

    if n_succ:
        ot = overtake[success].astype(np.int64)
        hist = np.bincount(ot, minlength=remaining + 1)[:remaining + 1]
        stats.update({
            "overtake_mean": round(float(ot.mean()), 2),
            "overtake_median": float(np.median(ot)),
            "overtake_p90": float(np.percentile(ot, 90)),
            "instant_pct": round(float((ot <= 1).mean() * 100.0), 1),
            "overtake_hist": [int(c) for c in hist],
        })
    else:
        stats.update({
            "overtake_mean": None, "overtake_median": None, "overtake_p90": None,
            "instant_pct": 0.0, "overtake_hist": [0] * (remaining + 1),
        })
    return stats

# ----------------------------------------------------------------- 参数合并/校验
def _merge_params(overrides):
    """用户高级参数覆盖默认值。"""
    p = dict(DEFAULT_PARAMS)
    if not isinstance(overrides, dict):
        return p
    for k in ("pit_lane_transit", "pit_lane_transit_scvsc", "pit_lane_transit_red",
              "pit_normal_prob", "pit_normal_mean", "pit_normal_std",
              "pit_abnormal_mean", "pit_abnormal_std", "lap_noise_std",
              "form_noise_std",
              "red_restart_gain", "red_restart_laps", "pit_reaction_laps",
              "n_sim"):
        if overrides.get(k) is not None:
            p[k] = float(overrides[k])
    off = dict(DEFAULT_PARAMS["compound_offset"])
    raw_off = overrides.get("compound_offset")
    if isinstance(raw_off, dict):
        for c in COMPOUNDS:
            if raw_off.get(c) is not None:
                off[c] = float(raw_off[c])
    p["compound_offset"] = off
    return p


def _check_int(name, v, lo, hi):
    try:
        iv = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是整数")
    if not (lo <= iv <= hi):
        raise ValueError(f"{name} 需在 {lo}~{hi} 之间")
    return iv


def _check_float(name, v, lo, hi):
    try:
        fv = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是数字")
    if not (lo <= fv <= hi):
        raise ValueError(f"{name} 需在 {lo}~{hi} 之间")
    return fv


# ----------------------------------------------------------------- 轮胎库存(正赛日套装)
# 官方规则口径 —— FIA 2025 F1 运动规则 Article 30(非冲刺周末,适用 2022–2025 摩纳哥):
#   30.2d)ii) 每车手 13 套干胎 = 2 硬(C3) + 3 中(C4) + 8 软(C5);
#   30.5i)   其中 1 套 Q3 专用软胎(Q3 前不得使用/交还;Q3 车手排位后交还 1 套软胎)
#            + 2 套正赛保护套(正赛前不得交还);其余 10 套中 P1/P2/P3 结束后各交还 2 套,
#            共 6 套上交 → 练习赛后剩 7 套进入排位+正赛(Q3 车手再交还 1 套,剩 6 套)。
#   30.5m)   摩纳哥特例: 正赛须至少使用 3 套胎(2025 起生效)且至少 2 种干地配方。
#   30.5c)   装上后驶出维修区即计为"已用"(故排位套带 6~7 圈磨损但仍可用于正赛)。
# 默认模板 = 模拟生成的"排位赛后常见状态"(Q2 淘汰者 P11–P15 口径,7 套可用):
#   练习赛大量长距离测试 → FP 长距离套磨损 20+ 圈,排位套 6~7 圈,均按赛制合理量级。
TIRE_SOURCES = ["新胎", "FP1", "FP2", "FP3", "Q1", "Q2", "Q3"]
RACE_SETS_MAX = 7

TIRE_TEMPLATE = [
    # ---- 已交还 6 套(P1/P2/P3 后各 2 套,30.5i)iii–v),锁定不可用) ----
    {"id": "R1", "compound": "C5", "source": "FP1", "wear": 9, "available": False, "locked": True},
    {"id": "R2", "compound": "C5", "source": "新胎", "wear": 0, "available": False, "locked": True},
    {"id": "R3", "compound": "C5", "source": "FP2", "wear": 23, "available": False, "locked": True},
    {"id": "R4", "compound": "C4", "source": "FP1", "wear": 12, "available": False, "locked": True},
    {"id": "R5", "compound": "C5", "source": "FP3", "wear": 7, "available": False, "locked": True},
    {"id": "R6", "compound": "C5", "source": "新胎", "wear": 0, "available": False, "locked": True},
    # ---- 正赛可用 7 套(排位赛后常见状态) ----
    {"id": "T1", "compound": "C5", "source": "新胎", "wear": 0, "available": True, "locked": False},   # Q3 专用套(Q1/Q2 淘汰者未使用)
    {"id": "T2", "compound": "C5", "source": "Q1", "wear": 6, "available": True, "locked": False},
    {"id": "T3", "compound": "C5", "source": "Q2", "wear": 7, "available": True, "locked": False},
    {"id": "T4", "compound": "C4", "source": "新胎", "wear": 0, "available": True, "locked": False},   # 正赛保护套
    {"id": "T5", "compound": "C4", "source": "新胎", "wear": 0, "available": True, "locked": False},
    {"id": "T6", "compound": "C3", "source": "新胎", "wear": 0, "available": True, "locked": False},   # 正赛保护套
    {"id": "T7", "compound": "C3", "source": "FP2", "wear": 21, "available": True, "locked": False},   # FP2 长距离测试后保留
]

# 官方规则摘要(随 meta 下发,供前端展示,避免"描述不清")
TIRE_RULES = {
    "allocation": "每车手 13 套干胎 = 2 硬(C3) + 3 中(C4) + 8 软(C5)(FIA 30.2d)ii))",
    "handback": "P1/P2/P3 结束后各交还 2 套(共 6 套);Q3 专用软胎 1 套 + 正赛保护套 2 套不得提前交还(FIA 30.5i))",
    "race_sets": "练习赛后可用 7 套;晋级 Q3 的车手排位后另交还 1 套软胎,正赛可用 6 套",
    "monaco_rule": "摩纳哥特例: 正赛至少使用 3 套胎(2025 起)+ 至少 2 种干地配方(FIA 30.5m))",
    "used_def": "装上后驶出维修区即计为已用:排位套带 6~7 圈磨损但可用于正赛(FIA 30.5c))",
    "red_flag": "红旗暂停期间可免费更换轮胎(FIA 57.4b)vii));恢复可采静态发车(FIA 58.11)",
}


def default_tires():
    return [dict(t) for t in TIRE_TEMPLATE]


def normalize_tires(raw):
    """校验/回退轮胎库存列表(正赛可用 ≤ RACE_SETS_MAX 套)。"""
    if raw is None:
        return default_tires()
    if not isinstance(raw, list) or not raw:
        raise ValueError("tires 必须是非空数组")
    out = []
    for i, t in enumerate(raw):
        if not isinstance(t, dict):
            raise ValueError(f"tires[{i}] 必须是对象")
        comp = str(t.get("compound", "")).strip().upper()
        if comp not in COMPOUNDS:
            raise ValueError(f"tires[{i}].compound 必须是 {COMPOUNDS} 之一")
        wear = _check_int(f"tires[{i}].wear", t.get("wear", 0), 0, TOTAL_LAPS - 1)
        locked = bool(t.get("locked", False))
        avail = t.get("available")
        avail = True if avail is None else bool(avail)
        if locked:                      # 练习套已上交,强制不可用
            avail = False
        src = str(t.get("source", "新胎")).strip() or "新胎"
        tid = str(t.get("id", f"T{i + 1}")).strip() or f"T{i + 1}"
        out.append({"id": tid, "compound": comp, "source": src,
                    "wear": wear, "available": avail, "locked": locked})
    n_avail = sum(1 for t in out if t["available"])
    if n_avail > RACE_SETS_MAX:
        raise ValueError(f"正赛可用轮胎不得超过 {RACE_SETS_MAX} 套"
                         f"(练习赛已用/上交 6 套,当前勾选 {n_avail} 套)")
    return out


def _tire_groups(tires):
    """按 (配方, 已用圈数) 聚合可互换的轮胎套装。"""
    groups, order = {}, []
    for t in tires:
        if not t.get("available", True):
            continue
        key = (t["compound"], t["wear"])
        if key not in groups:
            groups[key] = {"compound": t["compound"], "wear": t["wear"],
                           "count": 0, "sets": []}
            order.append(key)
        groups[key]["count"] += 1
        groups[key]["sets"].append(t)
    return [groups[k] for k in order]


def _best_wear_by_compound(tires):
    """每配方可用套装中最小磨损(全新优先)。"""
    out = {}
    for t in normalize_tires(tires):
        if not t["available"]:
            continue
        c = t["compound"]
        if c not in out or t["wear"] < out[c]:
            out[c] = t["wear"]
    return out


# ----------------------------------------------------------------- 库存约束 DP
PRERACE_PIT_LOSS = 19.0     # 与 脚本\solve_dp_strategy.py 的 PIT_LOSS 一致
MIN_STINT_LAPS = 5          # 每段最少圈数(同 DP 脚本)
MAX_TYRE_AGE = 50           # 单套轮胎最大使用圈数(同 DP 脚本 MAX_TYRE_AGE)


def _group_cost_tables(groups, driver, offsets):
    """每组的分段代价表: tab[L] = 该套装跑 L 圈的总秒数(L=0..78)。"""
    base = STORE.base_time(driver)
    tables = []
    for g in groups:
        params = STORE.deg_params(driver, g["compound"])
        tt = lap_profile(g["compound"], g["wear"], TOTAL_LAPS, base,
                         params, offsets)
        tables.append(np.concatenate([[0.0], np.cumsum(tt)]))
    return tables


def _solve_inventory(groups, tables, min_stops, max_stops=2):
    """库存约束下枚举求解,返回 (总秒数, 计划[(组号, 圈数), ...])。

    - 每组最多使用次数 = 库存套数;每段 ≥ MIN_STINT_LAPS 圈
    - 每次进站必须更换配方(与 脚本\solve_dp_strategy.py 的 DP 一致;
      相邻同配方双 stint 的"胎龄重置"套利无实战意义,禁用)
    - 每段结束胎龄 ≤ MAX_TYRE_AGE;至少使用 2 种配方(FIA 规则)
    """
    n = len(groups)
    comp_of = [g["compound"] for g in groups]
    maxl = [max(MIN_STINT_LAPS, MAX_TYRE_AGE - int(g["wear"])) for g in groups]
    total = TOTAL_LAPS
    best = {k: (float("inf"), None) for k in range(min_stops, max_stops + 1)}

    def diverse(idxs):
        return len({comp_of[i] for i in idxs}) >= 2

    # ---- 1 停: 两段 (i, j),换配方
    if 1 in best:
        for i in range(n):
            for j in range(n):
                if comp_of[i] == comp_of[j]:      # 进站必须换配方
                    continue
                if i == j and groups[i]["count"] < 2:
                    continue
                if not diverse([i, j]):
                    continue
                ci, cj = tables[i], tables[j]
                s_lo = MIN_STINT_LAPS
                s_hi = min(maxl[i], total - MIN_STINT_LAPS)
                for s in range(s_lo, s_hi + 1):
                    if total - s > maxl[j]:
                        continue
                    c = ci[s] + cj[total - s] + PRERACE_PIT_LOSS
                    if c < best[1][0]:
                        best[1] = (c, [(i, s), (j, total - s)])

    # ---- 2 停: 三段 (i, j, k),相邻换配方;前两段用最小加卷积
    if 2 in best:
        m_lo, m_hi = 2 * MIN_STINT_LAPS, total - MIN_STINT_LAPS
        m_arr = np.arange(m_lo, m_hi + 1)
        for i in range(n):
            for j in range(n):
                if comp_of[i] == comp_of[j]:      # 进站必须换配方
                    continue
                if i == j and groups[i]["count"] < 2:
                    continue
                ci, cj = tables[i], tables[j]
                b12 = np.full(total + 1, np.inf)
                a12 = np.zeros(total + 1, dtype=np.int64)
                for s in range(MIN_STINT_LAPS, min(maxl[i], total - 2 * MIN_STINT_LAPS) + 1):
                    lo, hi = max(s + MIN_STINT_LAPS, m_lo), m_hi
                    if lo > hi:
                        continue
                    cand = ci[s] + cj[lo - s: hi - s + 1]
                    seg = b12[lo: hi + 1]
                    better = cand < seg
                    # 第二段圈数不得超过该套装胎龄上限
                    m_rng = np.arange(lo, hi + 1)
                    better &= (m_rng - s) <= maxl[j]
                    if better.any():
                        idx = np.nonzero(better)[0]
                        b12[lo + idx] = cand[idx]
                        a12[lo + idx] = s
                for k in range(n):
                    if comp_of[j] == comp_of[k]:  # 进站必须换配方
                        continue
                    cnt = {}
                    for g in (i, j, k):
                        cnt[g] = cnt.get(g, 0) + 1
                    if any(groups[g]["count"] < c for g, c in cnt.items()):
                        continue
                    if not diverse([i, j, k]):
                        continue
                    ck = tables[k]
                    third = total - m_arr
                    ok = (third <= maxl[k]) & np.isfinite(b12[m_arr])
                    if not ok.any():
                        continue
                    totals = np.where(ok, b12[m_arr] + ck[np.clip(third, 0, total)]
                                      + 2 * PRERACE_PIT_LOSS, np.inf)
                    q = int(np.argmin(totals))
                    c = float(totals[q])
                    if c < best[2][0]:
                        m = int(m_arr[q])
                        s1 = int(a12[m])
                        best[2] = (c, [(i, s1), (j, m - s1), (k, total - m)])

    # 取允许范围内的最优
    feas = [(t, plan) for k, (t, plan) in best.items()
            if plan is not None and k >= min_stops]
    if not feas:
        return float("inf"), None
    return min(feas, key=lambda x: x[0])


def _plan_to_stints(plan, groups):
    """计划 → stint 明细(带所用套装标签)。"""
    stints, used = [], {}
    lap = 1
    for gi, laps in plan:
        g = groups[gi]
        idx = used.get(gi, 0)
        used[gi] = idx + 1
        t = g["sets"][min(idx, len(g["sets"]) - 1)]
        label = f"{g['compound']} · {t['source']}" + \
                (f" · 已用{g['wear']}圈" if g["wear"] else " · 新胎")
        stints.append({
            "compound": g["compound"], "laps": laps,
            "start": lap, "end": lap + laps - 1,
            "wear": g["wear"], "set_id": t.get("id", ""),
            "set_label": label, "set_source": t.get("source", ""),
        })
        lap += laps
    return stints


def _live_series(driver, stints, offsets):
    """按各段实际套装磨损生成逐圈圈速(进站圈附加进站损失)。"""
    base = STORE.base_time(driver)
    laps = []
    for st in stints:
        params = STORE.deg_params(driver, st["compound"])
        laps.extend(lap_profile(st["compound"], st["wear"], st["laps"],
                                base, params, offsets).tolist())
    pit_laps = [st["end"] for st in stints[:-1]]
    for pl in pit_laps:
        if 1 <= pl <= len(laps):
            laps[pl - 1] += PRERACE_PIT_LOSS
    return laps, pit_laps


def _strategy_text(stints):
    return " → ".join(f"{s['compound']}({s['laps']}圈)" for s in stints)


# ----------------------------------------------------------------- 轮胎选择
def tire_choice_etime(compound_options, remaining, code, offsets,
                      wear_by_compound=None):
    """各配方用库存最优套装(磨损最小)跑完剩余圈数的期望总用时(秒)。"""
    base = STORE.base_time(code)
    wear_by_compound = wear_by_compound or {}
    out = {}
    for comp in compound_options:
        params = STORE.deg_params(code, comp)
        w = int(wear_by_compound.get(comp, 0))
        out[comp] = round(float(lap_profile(comp, w, remaining, base,
                                            params, offsets).sum()), 1)
    return out

def _auto_pick_set(tires, remaining, code, offsets, base):
    """系统自动换胎: 从可用库存中挑选"跑完剩余圈数期望用时最短"的套装。"""
    best = None
    for t in tires:
        if not t.get("available", True):
            continue
        params = STORE.deg_params(code, t["compound"])
        et = float(lap_profile(t["compound"], t["wear"], max(remaining, 1),
                               base, params, offsets).sum())
        if best is None or et < best["etime_s"]:
            best = {"id": t["id"], "compound": t["compound"],
                    "source": t["source"], "wear": t["wear"],
                    "set_label": f"{t['compound']} · {t['source']}"
                                 f"{' · 已用' + str(t['wear']) + '圈' if t['wear'] else ' · 新胎'}",
                    "etime_s": round(et, 1)}
    return best


# ----------------------------------------------------------------- 对手换胎推断
def infer_rival_pit_status(rival_age, current_lap):
    """从对手当前胎龄推断其是否已进站换胎(启发式,供决策参考与 AI 分析)。

    原理: 轮胎"已用圈数"= 该套装装上后的行驶圈数。对手若从未进站,
    胎龄应≈比赛已进行圈数(发车胎);胎龄远小于赛程 → 该套装必是中途装上,
    可反推其最近一次进站圈号 ≈ 当前圈 − 胎龄。
    注意: 红旗/安全车下的免费换胎同样会重置胎龄,推断需结合旗种背景解读。
    """
    try:
        age = int(rival_age)
        lap = int(current_lap)
    except (TypeError, ValueError):
        return {"status": "不确定", "confidence": "低", "est_change_lap": None,
                "reasoning": "胎龄/圈数输入无效", "strategy_read": ""}
    est_change = lap - age                      # 该套装装上的圈号
    if age >= lap - 1:
        return {
            "status": "未换胎", "confidence": "高", "est_change_lap": None,
            "reasoning": f"对手胎龄 {age} 圈 ≈ 已进行 {lap} 圈,与发车胎一致,基本确定未进站",
            "strategy_read": ("对手大概率执行一停长首段或尚未启动停站窗口;"
                              "你先进站可逼其表态,但需防其反向延长做 Overcut。"),
        }
    if est_change <= 1:
        return {
            "status": "未换胎", "confidence": "中", "est_change_lap": None,
            "reasoning": f"胎龄 {age} 圈与赛程 {lap} 圈基本同步,应为发车胎",
            "strategy_read": "对手仍在首段,停站窗口未开启,可按既定节奏推进。",
        }
    if age <= 3 and lap >= 12:
        return {
            "status": "已换胎", "confidence": "高", "est_change_lap": max(est_change, 1),
            "reasoning": (f"胎龄仅 {age} 圈而比赛已进行 {lap} 圈,"
                          f"该套装约在第 {max(est_change, 1)} 圈装上 —— 刚进过站"),
            "strategy_read": ("对手刚完成进站且换上较新轮胎,大概率目标一停到底;"
                              "你若尚未进站,应尽快在本窗口跟进(Undercut 失效)或"
                              "利用其新胎暖胎期争取出站窗口。"),
        }
    if est_change >= 5:
        return {
            "status": "已换胎", "confidence": "中", "est_change_lap": max(est_change, 1),
            "reasoning": (f"胎龄 {age} 圈明显小于赛程 {lap} 圈,"
                          f"估计第 {max(est_change, 1)} 圈前后进站换过胎"),
            "strategy_read": ("对手已执行一次进站,剩余赛程预计一停到底;"
                              "可对比双方换上套装的新旧与配方,评估 Undercut/Overcut 空间。"),
        }
    return {
        "status": "不确定", "confidence": "低", "est_change_lap": max(est_change, 1),
        "reasoning": (f"胎龄 {age} 圈与赛程 {lap} 圈差距不大(推测换胎圈约第 "
                      f"{max(est_change, 1)} 圈),也可能是红旗/安全车下的免费换胎"),
        "strategy_read": "建议结合比赛进程(是否出过 SC/红旗)人工复核后再决策。",
    }


# ----------------------------------------------------------------- 场景模拟入口
def run_simulation(scenario, my_driver, my_compound, my_age,
                   rival_driver, rival_compound, rival_age,
                   current_lap, gap_s, sc_type="SC",
                   n_sim=None, params=None, seed=None,
                   my_fit_compound=None, my_fit_wear=0,
                   rival_fit_compound=None, rival_fit_wear=0,
                   tires=None, auto_fit=True):
    """
    Stackelberg 场景蒙特卡洛。返回两个响应分支的成功率与超越圈数统计。

    gap_s 语义: 我方领先对手的秒数(负值 = 我方落后)。
    分支语义与 脚本\solve_stackelberg.py 一致,旗种按赛制重组:
      undercut: 我方先进站 → 对手"跟进 / 不跟"
      overcut : 对手先进站 → 我方"留在赛道 / 跟进"
      sc      : SC/VSC 窗口(进站损失几乎相同,合并建模) → "进站 / 不进站"
      red     : 红旗(比赛暂停+免费换胎+静态发车) → "换胎 / 不换胎"
    成功率为连续概率: 每圈圈速 ~ N(确定性圈速, 车手级实测 σ²)。
    """
    scenario = (scenario or "").strip().lower()
    _SC_ALIAS = {"sc_vsc": "sc", "scvsc": "sc", "safety_car": "sc",
                 "red_flag": "red", "redflag": "red"}
    scenario = _SC_ALIAS.get(scenario, scenario)
    if not scenario:                      # 兼容旧客户端只传 sc_type
        st_raw = (sc_type or "").strip().upper().replace("-", "_")
        scenario = ("red" if st_raw in ("RED", "RED_FLAG") else
                    "sc" if st_raw in ("SC", "VSC", "SC_VSC", "SCVSC") else "")
    if scenario not in SCENARIOS:
        raise ValueError(f"未知场景: {scenario}")
    my_compound = (my_compound or "").strip().upper()
    rival_compound = (rival_compound or "").strip().upper()
    for name, comp in (("我方胎型", my_compound), ("对手胎型", rival_compound)):
        if comp not in COMPOUNDS:
            raise ValueError(f"{name} 必须是 {COMPOUNDS} 之一")
    my_driver = (my_driver or "").strip().upper()
    rival_driver = (rival_driver or "").strip().upper()
    if my_driver not in STORE.metrics and my_driver not in STORE.strategies:
        raise ValueError(f"无我方车手数据: {my_driver}")
    if rival_driver not in STORE.metrics and rival_driver not in STORE.strategies:
        raise ValueError(f"无对手车手数据: {rival_driver}")

    my_age = _check_int("我方胎龄", my_age, 0, TOTAL_LAPS - 1)
    rival_age = _check_int("对手胎龄", rival_age, 0, TOTAL_LAPS - 1)
    current_lap = _check_int("当前圈数", current_lap, 1, TOTAL_LAPS - 1)
    gap_s = _check_float("当前差距", gap_s, -120.0, 120.0)

    # 进站换上的轮胎套装(来自轮胎库存,练习/排位后的磨损状态)
    # auto_fit=True(默认): 由系统自动挑选最合理套装,忽略下方手动选择
    auto_fit = True if auto_fit is None else bool(auto_fit)
    if not auto_fit:
        my_fit_compound = (my_fit_compound or my_compound).strip().upper()
        rival_fit_compound = (rival_fit_compound or rival_compound).strip().upper()
        for name, comp in (("换上胎型(我方)", my_fit_compound),
                           ("换上胎型(对手)", rival_fit_compound)):
            if comp not in COMPOUNDS:
                raise ValueError(f"{name} 必须是 {COMPOUNDS} 之一")
        my_fit_wear = _check_int("换上磨损(我方)", my_fit_wear, 0, TOTAL_LAPS - 1)
        rival_fit_wear = _check_int("换上磨损(对手)", rival_fit_wear, 0, TOTAL_LAPS - 1)

    p = _merge_params(params)
    n_sim = _check_int("模拟次数", n_sim or p["n_sim"], 500, 200000)
    remaining = TOTAL_LAPS - current_lap
    rng = np.random.default_rng(None if seed in (None, "", 0, "0") else int(seed))
    offsets = p["compound_offset"]

    # 圈速波动 σ: 未手动指定时按车手实测标定(14_圈速波动估计.csv,近似正态)
    auto_noise = not (isinstance(params, dict) and params.get("lap_noise_std") is not None)
    if auto_noise:
        sigma_me, sigma_riv = driver_noise(my_driver), driver_noise(rival_driver)
        noise_mode = "auto"
    else:
        sigma_me = sigma_riv = max(0.0, float(p["lap_noise_std"]))
        noise_mode = "manual"
    form_std = max(0.0, float(p["form_noise_std"]))

    my_base = STORE.base_time(my_driver)
    rival_base = STORE.base_time(rival_driver)
    my_deg = STORE.deg_params(my_driver, my_compound)
    rival_deg = STORE.deg_params(rival_driver, rival_compound)

    # 系统自动换胎(auto_fit, 默认): 从轮胎库存中挑选"跑完剩余圈数期望用时最短"的套装
    tires = normalize_tires(tires)
    auto_info = None
    if auto_fit:
        my_pick = _auto_pick_set(tires, remaining, my_driver, offsets, my_base)
        riv_pick = _auto_pick_set(tires, remaining, rival_driver, offsets, rival_base)
        if my_pick:
            my_fit_compound, my_fit_wear = my_pick["compound"], my_pick["wear"]
        else:
            my_fit_compound, my_fit_wear = my_compound, my_age
        if riv_pick:
            rival_fit_compound, rival_fit_wear = riv_pick["compound"], riv_pick["wear"]
        else:
            rival_fit_compound, rival_fit_wear = rival_compound, rival_age
        auto_info = {"my": my_pick, "rival": riv_pick}

    # 进站通道损失: SC 与 VSC 几乎相同 → 合并一档;红旗 = 免费换胎(无损失)
    transit = p["pit_lane_transit"]
    if scenario == "sc":
        transit = p["pit_lane_transit_scvsc"]
        sc_type = "SC/VSC"
    elif scenario == "red":
        transit = p["pit_lane_transit_red"]
        sc_type = "RED"

    def profile(compound, age0, base, deg):
        return lap_profile(compound, age0, remaining, base, deg, offsets)

    # 反应延迟: 跟进方在对方进站后需 1~2 圈才完成响应(这正是 Undercut/Overcut
    # 的机制来源 —— 先进站者在"重叠圈"用新胎对旧胎赚取时间)
    react = max(0, int(p.get("pit_reaction_laps", 2) or 0))

    # 分支定义: (key, 标签, 说明, 我方进站, 对手进站, 我方胎龄, 对手胎龄, 我方延迟, 对手延迟)
    if scenario == "undercut":
        branch_defs = [
            ("follow", "对手跟进进站",
             f"对手观察 {react} 圈后跟进进站(旧胎跑重叠圈),双方最终都换新胎;"
             "你的收益 = 重叠圈里新胎对旧胎赚到的时间",
             True, True, 0, 0, 0, react),
            ("stay", "对手不跟",
             "对手留在赛道用旧胎跑完剩余赛程,你出站后落后但新胎优势需在赛道上兑现",
             True, False, 0, rival_age, 0, 0),
        ]
    elif scenario == "overcut":
        branch_defs = [
            ("stay", "你留在赛道",
             "对手先进站换胎,你利用干净空气继续跑旧胎,吃对手的进站损失+新胎暖胎期",
             False, True, my_age, 0, 0, 0),
            ("follow", "你跟进进站",
             f"你观察 {react} 圈后跟进进站(旧胎跑重叠圈),对手先换新胎已赚得重叠圈收益",
             True, True, my_age, 0, react, 0),
        ]
    elif scenario == "red":
        # 红旗 = 比赛暂停 + 免费换胎(无进站损失) + 静态发车(新/软胎起步占优)
        # Stackelberg: 你先决定换/不换;对手免费换胎是占优策略,默认跟随换胎
        branch_defs = [
            ("change", "红旗免费换胎",
             "比赛暂停期间免费换胎(零损失),静态发车用更新/更软的胎获取抓地增益;"
             "对手同样免费换胎,双方比拼换上套装的质量",
             True, True, 0, 0, 0, 0),
            ("stay", "红旗不换胎",
             "保留当前旧胎(为后段省下一套新胎),但对手免费换上新胎后,"
             "在静态发车的前几圈利用抓地优势直接得利",
             False, True, my_age, 0, 0, 0),
        ]
    else:  # sc (SC/VSC 合并)
        branch_defs = [
            ("pit", "你进站(廉价窗口)",
             f"SC/VSC 限速下进站损失大幅降低(仅 {transit:.1f}s 通道),换新胎跑完剩余 {remaining} 圈",
             True, False, 0, rival_age, 0, 0),
            ("stay", "你不进站",
             "留在赛道保位置,但对手进站损失小,可能被翻掉",
             False, False, my_age, rival_age, 0, 0),
        ]

    my_fit_deg = STORE.deg_params(my_driver, my_fit_compound)
    rival_fit_deg = STORE.deg_params(rival_driver, rival_fit_compound)

    def restart_bonus(compound, wear):
        """红旗静态发车抓地增益(秒/圈,负=更快): 越软越新的胎优势越大。"""
        soft = {"C5": 1.0, "C4": 0.6, "C3": 0.3}.get(compound, 0.5)
        fresh = max(0.0, 1.0 - float(wear) / 20.0)
        return p["red_restart_gain"] * soft * fresh

    def build_lt(pit, delay, cur_comp, cur_age, fit_comp, fit_wear, base, code):
        """构造逐圈圈速: 不进站=旧胎到底; 延迟进站=先旧胎跑重叠圈再换新胎。"""
        if not pit:
            return profile(cur_comp, cur_age, base, STORE.deg_params(code, cur_comp))
        if delay > 0 and delay < remaining:
            pre = profile(cur_comp, cur_age, base,
                          STORE.deg_params(code, cur_comp))[:delay]
            post = profile(fit_comp, fit_wear, base,
                           STORE.deg_params(code, fit_comp))
            return np.concatenate([pre, post])[:remaining]
        return profile(fit_comp, fit_wear, base,
                       STORE.deg_params(code, fit_comp))

    branches = []
    for key, label, desc, me_pit, riv_pit, my_a0, riv_a0, dly_me, dly_riv in branch_defs:
        # 进站分支: 换上所选库存套装(含练习/排位磨损);延迟进站方先用旧胎跑重叠圈
        lt_me = build_lt(me_pit, dly_me, my_compound,
                         my_age if (me_pit and dly_me) else my_a0,
                         my_fit_compound, my_fit_wear, my_base, my_driver)
        lt_riv = build_lt(riv_pit, dly_riv, rival_compound,
                          rival_age if (riv_pit and dly_riv) else riv_a0,
                          rival_fit_compound, rival_fit_wear, rival_base,
                          rival_driver)
        t0_me = (sample_pit_times(rng, n_sim, transit, p) if me_pit
                 else np.zeros(n_sim))
        t0_riv = (sample_pit_times(rng, n_sim, transit, p) + gap_s if riv_pit
                  else np.full(n_sim, float(gap_s)))

        # 红旗: 静态发车新/软胎抓地增益作用于重启后前 N 圈
        restart_gain = None
        if scenario == "red":
            gain_me = restart_bonus(*(my_fit_compound, my_fit_wear) if me_pit
                                    else (my_compound, my_age))
            gain_riv = restart_bonus(*(rival_fit_compound, rival_fit_wear) if riv_pit
                                     else (rival_compound, rival_age))
            n_gain = max(0, int(p["red_restart_laps"]))
            if n_gain and remaining > 0:
                k = min(n_gain, remaining)
                lt_me = lt_me.copy()
                lt_riv = lt_riv.copy()
                lt_me[:k] -= gain_me
                lt_riv[:k] -= gain_riv
            restart_gain = {"my": round(gain_me, 3), "rival": round(gain_riv, 3),
                            "net": round(gain_me - gain_riv, 3),  # 正=我方得利
                            "laps": min(n_gain, remaining)}

        success, overtake, overtaken, swapped, delta = _run_branch(
            rng, n_sim, remaining, lt_me, lt_riv, t0_me, t0_riv,
            sigma_me, sigma_riv, form_std, form_std)
        stats = _branch_stats(success, overtake, overtaken, swapped, delta, remaining)
        med = stats["overtake_median"]
        if med is None:
            pos_note = "该分支无成功样本"
        elif med <= 0:
            pos_note = "无需追赶 — 位置已保持/经进站窗口直接获得"
        else:
            pos_note = (f"平均 {stats['overtake_mean']:.1f} 圈后完成超越"
                        f"(中位 {med:.0f} 圈)")
        branches.append({
            "key": key, "label": label, "desc": desc,
            "my_pit": me_pit, "rival_pit": riv_pit,
            "pos_note": pos_note,
            "restart_gain": restart_gain,
            "cfg": [me_pit, riv_pit, my_a0, riv_a0, dly_me, dly_riv],
            **stats,
        })

    # ---- 不动基线: 双方均不进站,用于剥离"车速差异"看"策略本身值多少"
    lt_me0 = profile(my_compound, my_age, my_base, my_deg)
    lt_riv0 = profile(rival_compound, rival_age, rival_base, rival_deg)
    s0, _, _, _, d0 = _run_branch(
        rng, n_sim, remaining, lt_me0, lt_riv0,
        np.zeros(n_sim), np.full(n_sim, float(gap_s)),
        sigma_me, sigma_riv, form_std, form_std)
    baseline_rate = round(float(s0.mean()) * 100.0, 2)
    for b in branches:
        b["baseline_rate"] = baseline_rate
        b["strategy_gain_pp"] = round(b["success_rate"] - baseline_rate, 2)

    best = max(branches, key=lambda b: b["success_rate"])

    # ---- 鲁棒性检验: 各分支在关键参数扰动下的成功率区间(小样本复算)
    def _robust_rate(cfg, mod_deg=1.0, base_shift=0.0, gap_shift=0.0,
                     noise_mult=1.0, n=3000, seed=20261001):
        rng2 = np.random.default_rng(seed)
        me_pit, riv_pit, my_a0, riv_a0, dly_me, dly_riv = cfg

        def prof2(comp, age, base, code):
            dp = dict(STORE.deg_params(code, comp))
            dp["deg"] = dp["deg"] * mod_deg
            return profile(comp, age, base + base_shift, dp)

        def build2(pit, delay, cur_comp, cur_age, fit_comp, fit_wear, base, code):
            if not pit:
                return prof2(cur_comp, cur_age, base, code)
            if delay > 0 and delay < remaining:
                pre = prof2(cur_comp, cur_age, base, code)[:delay]
                post = prof2(fit_comp, fit_wear, base, code)
                return np.concatenate([pre, post])[:remaining]
            return prof2(fit_comp, fit_wear, base, code)

        lt_me = build2(me_pit, dly_me, my_compound,
                       my_age if (me_pit and dly_me) else my_a0,
                       my_fit_compound, my_fit_wear, my_base, my_driver)
        lt_riv = build2(riv_pit, dly_riv, rival_compound,
                        rival_age if (riv_pit and dly_riv) else riv_a0,
                        rival_fit_compound, rival_fit_wear, rival_base,
                        rival_driver)
        if me_pit:
            t0_me2 = sample_pit_times(rng2, n, transit, p)
        else:
            t0_me2 = np.zeros(n)
        if riv_pit:
            t0_riv2 = sample_pit_times(rng2, n, transit, p) + (gap_s + gap_shift)
        else:
            t0_riv2 = np.full(n, gap_s + gap_shift)
        s, *_ = _run_branch(rng2, n, remaining, lt_me, lt_riv, t0_me2, t0_riv2,
                            sigma_me * noise_mult, sigma_riv * noise_mult,
                            form_std * noise_mult, form_std * noise_mult)
        return round(float(s.mean()) * 100.0, 2)

    rob_cases = [
        ("名义(复核)", {}),
        ("衰减斜率 −20%", {"mod_deg": 0.8}),
        ("衰减斜率 +20%", {"mod_deg": 1.2}),
        ("我方圈速 −0.3s/圈", {"base_shift": -0.3}),
        ("我方圈速 +0.3s/圈", {"base_shift": 0.3}),
        ("差距 −2s(更落后)", {"gap_shift": -2.0}),
        ("差距 +2s(更领先)", {"gap_shift": 2.0}),
        ("波动 σ ×1.5", {"noise_mult": 1.5}),
    ]
    for b in branches:
        rl = [{"name": nm, "success_rate": _robust_rate(b["cfg"], **kw)}
              for nm, kw in rob_cases]
        rates = [c["success_rate"] for c in rl]
        b["robustness"] = {"min": min(rates), "max": max(rates),
                           "range": round(max(rates) - min(rates), 1)}
        b["_rob_cases"] = rl
    robustness = {
        "branch": best["label"], "cases": best.get("_rob_cases") or [],
        "nominal": best["success_rate"],
        "min": best["robustness"]["min"], "max": best["robustness"]["max"],
        "range": best["robustness"]["range"],
    }

    # ---- 圈速差拆解(为何胜率高/低): 基准圈速差 + 换上套装期望差
    pace_note = (
        f"基准圈速: 我方 {my_base:.2f}s vs 对手 {rival_base:.2f}s"
        f"(差 {rival_base - my_base:+.2f}s/圈);"
        f"当前胎: {my_compound}@{my_age} vs {rival_compound}@{rival_age};"
        f"换上: 我方 {my_fit_compound}(磨损{my_fit_wear}圈) vs "
        f"对手 {rival_fit_compound}(磨损{rival_fit_wear}圈)")

    rec = {"branch": best["key"], "label": best["label"],
           "success_rate": best["success_rate"],
           "baseline_rate": baseline_rate,
           "strategy_gain_pp": best["strategy_gain_pp"],
           "risk_level": best["risk_level"],
           "relost_rate": best["relost_rate"],
           "swap_rate": best["swap_rate"],
           "robustness": {"min": robustness["min"], "max": robustness["max"],
                          "range": robustness["range"]}}
    risk_txt = (f"被反超风险{best['risk_level']}"
                f"(曾领先后得而复失 {best['relost_rate']:.1f}%)")
    gain_txt = f"(不动基线 {baseline_rate:.1f}%,策略增益 {best['strategy_gain_pp']:+.1f}pp)"
    if best["overtake_mean"] is not None and (best["overtake_median"] or 0) > 0:
        rec["text"] = (f"推荐「{best['label']}」: 成功率 {best['success_rate']:.1f}%{gain_txt},"
                       f"成功时约 {best['overtake_mean']:.1f} 圈完成超越;{risk_txt}")
    else:
        rec["text"] = (f"推荐「{best['label']}」: 成功率 {best['success_rate']:.1f}%{gain_txt},"
                       f"成功时可直接保持/获得位置;{risk_txt}")

    rule_note = None
    if scenario == "red":
        rule_note = (
            "红旗规则要点: ① 比赛暂停,可在车房内免费换胎(无进站时间损失);"
            f"② 恢复后为静态发车,换上更软/更新的胎可在前 "
            f"{int(p['red_restart_laps'])} 圈获得起步抓地增益;"
            "③ 若此前只用过一种干地配方,红旗免费换胎同时是满足"
            "\"正赛必须使用两种干地配方\"规则的绝佳机会。")

    return {
        "scenario": scenario,
        "scenario_name": SCENARIOS[scenario]["title"],
        "sc_type": sc_type if scenario in ("sc", "red") else None,
        "n_sim": n_sim,
        "current_lap": current_lap,
        "remaining": remaining,
        "gap_s": gap_s,
        "rule_note": rule_note,
        "baseline": {"label": "双方均不进站(基线)", "success_rate": baseline_rate},
        "pace_note": pace_note,
        "robustness": robustness,
        "inputs": {
            "my": {"driver": my_driver,
                   "name": DRIVER_NAMES.get(my_driver, my_driver),
                   "compound": my_compound, "age": my_age,
                   "fit_compound": my_fit_compound, "fit_wear": my_fit_wear,
                   "base_time": round(my_base, 2)},
            "rival": {"driver": rival_driver,
                      "name": DRIVER_NAMES.get(rival_driver, rival_driver),
                      "compound": rival_compound, "age": rival_age,
                      "fit_compound": rival_fit_compound,
                      "fit_wear": rival_fit_wear,
                      "base_time": round(rival_base, 2)},
        },
        "branches": branches,
        "recommendation": rec,
        "rival_pit_inference": infer_rival_pit_status(rival_age, current_lap),
        "fit_mode": "auto" if auto_fit else "manual",
        "auto_fit": auto_info,
        "tire_choice": tire_choice_etime(COMPOUNDS, remaining, my_driver,
                                         offsets, _best_wear_by_compound(tires)),
        "tire_choice_wear": _best_wear_by_compound(tires),
        "params_used": {
            "pit_lane_transit": transit,
            "pit_normal_prob": p["pit_normal_prob"],
            "pit_normal_mean": p["pit_normal_mean"],
            "pit_abnormal_mean": p["pit_abnormal_mean"],
            "compound_offset": offsets,
            "lap_noise_std": None if auto_noise else round(float(p["lap_noise_std"]), 3),
            "my_sigma": sigma_me,
            "rival_sigma": sigma_riv,
            "noise_mode": noise_mode,
            "form_noise_std": round(float(p["form_noise_std"]), 3),
            "red_restart_gain": p["red_restart_gain"] if scenario == "red" else None,
            "red_restart_laps": int(p["red_restart_laps"]) if scenario == "red" else None,
            "pit_reaction_laps": int(p["pit_reaction_laps"]),
        },
    }

# ----------------------------------------------------------------- 赛前模拟(问题一)
PIT_TOTAL_DEFAULT = DEFAULT_PARAMS["pit_lane_transit"] + (
    DEFAULT_PARAMS["pit_normal_prob"] * DEFAULT_PARAMS["pit_normal_mean"]
    + (1 - DEFAULT_PARAMS["pit_normal_prob"]) * DEFAULT_PARAMS["pit_abnormal_mean"])


def _mode_lap_series(driver, strategy, offsets):
    """按 DP 策略生成逐圈圈速(含进站损失),并缩放到与 DP 总用时一致。

    返回 (lap_times, scale, pit_laps): lap_times[k] 为第 k+1 圈总耗时。
    """
    stints = strategy["stints"]
    stops = max(len(stints) - 1, 0)
    base = STORE.base_time(driver)
    laps = []
    for st in stints:
        params = STORE.deg_params(driver, st["compound"])
        laps.extend(lap_profile(st["compound"], 0, st["laps"],
                                base, params, offsets).tolist())
    running = sum(laps)
    target = max(strategy["total_s"] - stops * PIT_TOTAL_DEFAULT, 1.0)
    scale = target / running if running > 0 else 1.0
    lap_times = [t * scale for t in laps]
    pit_laps = [st["end"] for st in stints[:-1]]      # 进站发生在该 stint 末圈
    for pl in pit_laps:
        if 1 <= pl <= len(lap_times):
            lap_times[pl - 1] += PIT_TOTAL_DEFAULT
    return lap_times, scale, pit_laps


def _fmt_hms(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}"


def _solve_mode(driver, groups, tables, mode, offsets):
    """按模式在库存约束下求解; 不可行返回 None。

    normal   = 恰好 1 停(2022–2024 常规规则实战常态,对比基准);
    two_stop = 恰好 2 停(亦为摩纳哥 2025 新规"至少 3 套胎"的求解口径)。
    """
    stops = 1 if mode == "normal" else 2
    total, plan = _solve_inventory(groups, tables, stops, max_stops=stops)
    if plan is None:
        return None
    stints = _plan_to_stints(plan, groups)
    lap_times, pit_laps = _live_series(driver, stints, offsets)
    cum, s = [], 0.0
    for t in lap_times:
        s += t
        cum.append(round(s, 2))
    return {"total_s": round(total, 1), "stints": stints,
            "lap_times": [round(t, 3) for t in lap_times],
            "cumulative": cum, "pit_laps": pit_laps}


def get_prerace(driver, mode="normal", tires=None):
    """车手赛前策略: 轮胎库存约束下实时 DP 求解。

    tires: 轮胎库存([{id, compound, source, wear, available}], None=默认模板)。
    每段分配具体库存套装,已用圈数作为该段起始胎龄;
    同时输出"全全新胎基线"以量化练习/排位磨损的时间代价。
    """
    driver = (driver or "").strip().upper()
    mode = (mode or "normal").strip().lower()
    if mode not in ("normal", "two_stop"):
        raise ValueError("mode 必须是 normal 或 two_stop")
    if driver not in STORE.metrics and driver not in STORE.strategies:
        raise ValueError(f"无该车手数据: {driver}")

    tires = normalize_tires(tires)
    offsets = DEFAULT_PARAMS["compound_offset"]
    groups = _tire_groups(tires)
    if not groups:
        raise ValueError("轮胎库存为空: 请至少勾选一套可用轮胎")
    tables = _group_cost_tables(groups, driver, offsets)

    main = _solve_mode(driver, groups, tables, mode, offsets)
    if main is None:
        raise ValueError("当前轮胎库存无法满足约束(至少 2 种配方,且套数够分段)")
    alt_mode = "two_stop" if mode == "normal" else "normal"
    alt = _solve_mode(driver, groups, tables, alt_mode, offsets)

    # 全新胎基线: 量化练习/排位磨损带来的总时间代价
    fresh_groups = _tire_groups([dict(t, wear=0) for t in tires])
    fresh_tables = _group_cost_tables(fresh_groups, driver, offsets)
    fresh = _solve_mode(driver, fresh_groups, fresh_tables, mode, offsets)
    tire_penalty = (round(main["total_s"] - fresh["total_s"], 1)
                    if fresh else None)

    # 分段明细(带套装标签)
    base = STORE.base_time(driver)
    stints_out, idx = [], 0
    for st in main["stints"]:
        params = STORE.deg_params(driver, st["compound"])
        seg = main["lap_times"][idx: idx + st["laps"]]
        raw = lap_profile(st["compound"], st["wear"], st["laps"], base,
                          params, offsets)
        stints_out.append({
            "compound": st["compound"], "laps": st["laps"],
            "start": st["start"], "end": st["end"],
            "wear": st["wear"], "set_id": st.get("set_id", ""),
            "set_label": st.get("set_label", ""),
            "stint_time": round(sum(seg), 1),
            "avg_lap": round(sum(seg) / max(st["laps"], 1), 3),
            "start_lap_time": round(seg[0], 3),
            "end_lap_time": round(seg[-1], 3),
            "raw_avg": round(float(raw.mean()), 3),
        })
        idx += st["laps"]

    main_series = {"lap_times": main["lap_times"],
                   "cumulative": main["cumulative"],
                   "pit_laps": main["pit_laps"]}
    alt_series = ({"lap_times": alt["lap_times"],
                   "cumulative": alt["cumulative"],
                   "pit_laps": alt["pit_laps"]} if alt else None)

    # 正常 vs 两停(均为库存约束实时解)
    if mode == "normal":
        normal_s, two_s = main["total_s"], (alt["total_s"] if alt else None)
        normal_text = _strategy_text(main["stints"])
        two_text = _strategy_text(alt["stints"]) if alt else None
    else:
        normal_s = alt["total_s"] if alt else None
        two_s = main["total_s"]
        normal_text = _strategy_text(alt["stints"]) if alt else None
        two_text = _strategy_text(main["stints"])
    cmp_row = {
        "normal_text": normal_text, "normal_s": normal_s,
        "two_text": two_text, "two_s": two_s,
        "delta_s": (round(two_s - normal_s, 1)
                    if (normal_s is not None and two_s is not None) else None),
        "note": ("纯时间口径下两停接近甚至略优,源于线性衰减模型的\"胎龄重置\"边际收益;"
                 "但摩纳哥超车极难、赛道位置价值远超该边际差异,"
                 "2022–2024 实战几乎全员一停,2025 起新规则强制至少 3 套胎。"
                 if (delta := (round(two_s - normal_s, 1)
                               if (normal_s is not None and two_s is not None) else None))
                 is not None and delta <= 3 else
                 "两停额外一次进站损失大于轮胎收益,验证了摩纳哥一停的常规性。"),
    }

    m = STORE.metrics.get(driver, {})
    return {
        "driver": driver,
        "name": m.get("name", DRIVER_NAMES.get(driver, driver)),
        "style": m.get("style", "—"),
        "base_time": m.get("base_time", 75.5),
        "mode": mode,
        "mode_name": ("最优策略(常规规则: 至少 2 种配方)" if mode == "normal"
                      else "强制两停(摩纳哥 2025 新规: 至少 3 套胎)"),
        "rule_note": ("规则约束(FIA 30.5m): 至少使用 2 种干地配方;摩纳哥自 2025 起"
                      "正赛须至少使用 3 套胎(即至少 2 次进站),「强制两停」模式即按该"
                      "新规求解。「最优策略」为 2022–2024 常规规则下的最优解。"),
        "strategy_text": _strategy_text(main["stints"]),
        "stops": len(main["stints"]) - 1,
        "total_s": main["total_s"],
        "total_display": _fmt_hms(main["total_s"]),
        "pit_laps": main["pit_laps"],
        "stints": stints_out,
        "series": main_series,
        "alt": {
            "mode": alt_mode,
            "mode_name": "强制两停" if alt_mode == "two_stop" else "最优策略(正常)",
            "strategy_text": _strategy_text(alt["stints"]) if alt else None,
            "total_s": alt["total_s"] if alt else None,
            "series": alt_series,
        },
        "compare": cmp_row,
        "tire_penalty_s": tire_penalty,
        "fresh_total_s": fresh["total_s"] if fresh else None,
        "tires": tires,
        "charts": {
            "all_drivers": "/charts/08b_车手最优策略对比.png",
            "two_stop": "/charts/08c_车手强制两停策略.png",
            "timeline": "/charts/08_最优停站策略时间线.png",
            "cumulative": "/charts/09_策略累计时间对比.png",
        },
    }


def _prerace_from_csv(driver, mode):
    """旧版: 读取预计算 CSV 策略(无轮胎库存概念),保留备查。"""
    driver = (driver or "").strip().upper()
    mode = (mode or "normal").strip().lower()
    if mode not in ("normal", "two_stop"):
        raise ValueError("mode 必须是 normal 或 two_stop")
    if driver not in STORE.strategies:
        raise ValueError(f"该车手无 DP 策略数据: {driver}")

    offsets = DEFAULT_PARAMS["compound_offset"]
    chosen = STORE.strategies[driver][mode]
    other_mode = "two_stop" if mode == "normal" else "normal"
    other = STORE.strategies[driver].get(other_mode)

    def series(strategy):
        if not strategy or not strategy["stints"]:
            return None
        lap_times, scale, pit_laps = _mode_lap_series(driver, strategy, offsets)
        cum, s = [], 0.0
        for t in lap_times:
            s += t
            cum.append(round(s, 2))
        return {"lap_times": [round(t, 3) for t in lap_times],
                "cumulative": cum, "pit_laps": pit_laps}

    main_series = series(chosen)
    alt_series = series(other)

    # 分段明细
    stints_out = []
    if main_series:
        scale = 1.0
        base = STORE.base_time(driver)
        idx = 0
        for st in chosen["stints"]:
            params = STORE.deg_params(driver, st["compound"])
            raw = lap_profile(st["compound"], 0, st["laps"], base, params, offsets)
            lap_times = main_series["lap_times"][idx: idx + st["laps"]]
            stints_out.append({
                "compound": st["compound"],
                "laps": st["laps"],
                "start": st["start"],
                "end": st["end"],
                "stint_time": round(sum(lap_times), 1),
                "avg_lap": round(sum(lap_times) / max(st["laps"], 1), 3),
                "start_lap_time": round(float(lap_times[0]), 3),
                "end_lap_time": round(float(lap_times[-1]), 3),
                "raw_avg": round(float(raw.mean()), 3),
            })
            idx += st["laps"]

    cmp_row = STORE.compare.get(driver, {})
    m = STORE.metrics.get(driver, {})
    return {
        "driver": driver,
        "name": m.get("name", DRIVER_NAMES.get(driver, driver)),
        "style": m.get("style", "—"),
        "base_time": m.get("base_time", 75.5),
        "mode": mode,
        "mode_name": "最优策略(正常)" if mode == "normal" else "强制两停",
        "strategy_text": chosen["text"],
        "stops": chosen["stops"],
        "total_s": chosen["total_s"],
        "total_display": chosen["total_display"],
        "pit_laps": main_series["pit_laps"] if main_series else [],
        "stints": stints_out,
        "series": main_series,
        "alt": {
            "mode": other_mode,
            "mode_name": "强制两停" if other_mode == "two_stop" else "最优策略(正常)",
            "strategy_text": other["text"] if other else None,
            "total_s": other["total_s"] if other else None,
            "series": alt_series,
        },
        "compare": cmp_row,
        "charts": {
            "all_drivers": "/charts/08b_车手最优策略对比.png",
            "two_stop": "/charts/08c_车手强制两停策略.png",
            "timeline": "/charts/08_最优停站策略时间线.png",
            "cumulative": "/charts/09_策略累计时间对比.png",
        },
    }

# ----------------------------------------------------------------- AI 策略分析
# 接入外部大模型 API(OpenAI 兼容 /chat/completions),可选:
#   配置来源(环境变量优先,其次 web/ai_config.local.json,后者不入库):
#     F1_AI_API_KEY / api_key    必填 —— API 密钥(未配置时自动降级为本地规则分析)
#     F1_AI_BASE_URL / base_url  可选 —— 默认 https://api.deepseek.com
#     F1_AI_MODEL / model        可选 —— 默认 deepseek-flash(DeepSeek-V4.1-Flash)
# 无论哪种来源,返回值都会明确标注,绝不冒充 AI 输出。
def _ai_config():
    key = (os.environ.get("F1_AI_API_KEY") or "").strip()
    base = (os.environ.get("F1_AI_BASE_URL") or "").strip()
    model = (os.environ.get("F1_AI_MODEL") or "").strip()
    local_path = os.path.join(WEB_DIR, "ai_config.local.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                local = json.load(f)
        except (OSError, json.JSONDecodeError):
            local = {}
        key = key or str(local.get("api_key") or "").strip()
        base = base or str(local.get("base_url") or "").strip()
        model = model or str(local.get("model") or "").strip()
    base = base or "https://api.deepseek.com"
    model = model or "deepseek-flash"
    return key, base, model


def _result_digest(result):
    """把模拟结果压缩成 LLM /本地分析共用的事实描述。"""
    lines = []
    my, rv = result.get("inputs", {}).get("my", {}), result.get("inputs", {}).get("rival", {})
    lines.append(f"场景: {result.get('scenario_name')}"
                 + (f"(旗种 {result.get('sc_type')})" if result.get("sc_type") else ""))
    lines.append(f"我方 {my.get('name')}({my.get('compound')},胎龄{my.get('age')}) "
                 f"vs 对手 {rv.get('name')}({rv.get('compound')},胎龄{rv.get('age')})")
    lines.append(f"第 {result.get('current_lap')} 圈,剩余 {result.get('remaining')} 圈,"
                 f"与前车差距 {result.get('gap_s')}s(正=领先)")
    lines.append(f"进站换上: 我方 {my.get('fit_compound')}(磨损{my.get('fit_wear')}圈),"
                 f"对手 {rv.get('fit_compound')}(磨损{rv.get('fit_wear')}圈)")
    for b in result.get("branches", []):
        lines.append(
            f"分支[{b.get('label')}]: 成功率 {b.get('success_rate')}%"
            f"(不动基线 {b.get('baseline_rate')}%,策略增益 {b.get('strategy_gain_pp'):+}pp),"
            f"被反超风险等级 {b.get('risk_level')}"
            f"(曾领先后得而复失 {b.get('relost_rate')}%),"
            f"期望完赛时间差 {b.get('mean_finish_delta')}s,"
            f"{b.get('pos_note') or ''}")
    if result.get("pace_note"):
        lines.append(f"圈速差拆解: {result['pace_note']}")
    rb = result.get("robustness") or {}
    if rb:
        lines.append(f"鲁棒性: 最优分支「{rb.get('branch')}」名义 {rb.get('nominal')}%,"
                     f"关键参数扰动下区间 {rb.get('min')}%~{rb.get('max')}%"
                     f"(跨度 {rb.get('range')}pp,越窄越可靠)")
    pu = result.get("params_used", {}) or {}
    lines.append(f"圈速波动 σ: 我方 {pu.get('my_sigma')}s / 对手 {pu.get('rival_sigma')}s"
                 f"({pu.get('noise_mode')},实测标定近似正态)")
    inf = result.get("rival_pit_inference") or {}
    if inf:
        lines.append(f"对手换胎推断: {inf.get('status')}(置信度{inf.get('confidence')});"
                     f"{inf.get('reasoning', '')};战略解读: {inf.get('strategy_read', '')}")
    if result.get("rule_note"):
        lines.append(str(result["rule_note"]))
    return "\n".join(lines)


def _llm_analyze(base, model, key, digest):
    import urllib.request
    prompt = (
        "你是一位资深 F1 策略工程师,正在摩纳哥大奖赛的维修墙工作。"
        "请基于以下蒙特卡洛模拟结果(成功率与被反超风险均为圈速正态波动下的概率),"
        "给出简明的中文策略分析:① 推荐哪个分支及理由;② 每个分支的被反超风险如何解读;"
        "③ 轮胎与规则层面的注意事项;④ 一句话结论。控制在 250 字以内。\n\n"
        + digest)
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def _local_analysis(result):
    """本地规则分析: 由本次模拟数字按规则模板生成(非大模型输出)。"""
    branches = result.get("branches", [])
    if not branches:
        return "无分支结果,无法分析。"
    best = max(branches, key=lambda b: b.get("success_rate") or 0)
    worst = min(branches, key=lambda b: b.get("success_rate") or 0)
    out = []
    out.append(f"【结论】推荐「{best['label']}」(成功率 {best['success_rate']}%,"
               f"被反超风险{best['risk_level']})。")
    gap = (best.get("success_rate") or 0) - (worst.get("success_rate") or 0)
    if gap < 5:
        out.append(f"两分支成功率仅差 {gap:.1f} 个百分点,几乎持平 —— "
                   "圈速波动下两种选择的期望收益接近,可结合风险偏好取舍。")
    else:
        out.append(f"「{best['label']}」比「{worst['label']}」成功率高 {gap:.1f} 个百分点,"
                   "优势主要来自" + ("进站窗口的时间收益" if best.get("my_pit")
                   else "留在赛道的既有时差与干净空气") + "。")
    for b in branches:
        rl = b.get("risk_level")
        gain = b.get("strategy_gain_pp")
        gain_txt = f",策略增益 {gain:+.1f}pp(不动基线 {b.get('baseline_rate')}%)" if gain is not None else ""
        if rl == "高":
            out.append(f"⚠ 「{b['label']}」被反超风险高: 曾领先后得而复失的概率约 "
                       f"{b.get('relost_rate')}%,位置翻转频繁度 {b.get('swap_rate')}%{gain_txt},"
                       "得手后需重点防守。")
        elif rl == "中":
            out.append(f"「{b['label']}」被反超风险中等(得而复失 {b.get('relost_rate')}%"
                       f",翻转 {b.get('swap_rate')}%){gain_txt},需关注后段轮胎衰减与对手进站窗口。")
        else:
            out.append(f"「{b['label']}」被反超风险低(得而复失 {b.get('relost_rate')}%"
                       f",翻转 {b.get('swap_rate')}%),位置基本稳固{gain_txt}。")
    pu = result.get("params_used", {}) or {}
    out.append(f"概率口径: 每圈圈速 ~ N(确定性圈速, σ²),我方 σ={pu.get('my_sigma')}s,"
               f"对手 σ={pu.get('rival_sigma')}s(实测标定),故成功率是连续概率而非硬判定。")
    inf = result.get("rival_pit_inference") or {}
    if inf:
        out.append(f"对手换胎推断({inf.get('status')},置信度{inf.get('confidence')}): "
                   f"{inf.get('reasoning', '')}。{inf.get('strategy_read', '')}")
    if result.get("rule_note"):
        out.append("规则提示: " + str(result["rule_note"]))
    return "\n".join(out)


def ai_analysis(result):
    """策略 AI 分析入口。配置了 F1_AI_API_KEY 时调用大模型,否则本地规则分析。"""
    if not isinstance(result, dict) or "branches" not in result:
        raise ValueError("result 必须是 /api/simulate 的返回结果")
    digest = _result_digest(result)
    key, base, model = _ai_config()
    if key:
        try:
            text = _llm_analyze(base, model, key, digest)
            return {"source": "llm", "model": model, "enabled": True,
                    "text": text, "note": f"由大模型 {model} 生成"}
        except Exception as e:  # noqa: BLE001
            note = f"AI API 调用失败({e}),已降级为本地规则分析"
    else:
        note = ("未配置环境变量 F1_AI_API_KEY,当前为本地规则分析;"
                "配置后自动切换为大模型分析")
    return {"source": "local", "model": None, "enabled": False,
            "text": _local_analysis(result), "note": note}


# ----------------------------------------------------------------- 历史记录
def append_history(record):
    """追加一条历史记录(jsonl),返回带时间戳的完整记录。"""
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
    with _HISTORY_LOCK:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def read_history(limit=200):
    """按时间倒序读取历史记录。"""
    if not os.path.exists(HISTORY_FILE):
        return []
    with _HISTORY_LOCK:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    recs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    recs.reverse()
    return recs[: max(1, int(limit))]


def clear_history():
    with _HISTORY_LOCK:
        open(HISTORY_FILE, "w", encoding="utf-8").close()


# ----------------------------------------------------------------- 元数据
# 数据总览表(站内表格页,替代静态图片跳转)
TABLE_SPECS = [
    ("strategy", "车手最优(正常)策略对比表", "10b_车手策略对比表.csv"),
    ("two_stop", "车手强制两停策略对比表", "10c_车手强制两停对比表.csv"),
    ("compare", "正常 vs 强制两停对比表", "10d_正常vs两停对比表.csv"),
    ("deg", "车手轮胎衰减参数表", "07b_车手衰减参数表.csv"),
    ("style", "车手驾驶习惯指标表", "07c_车手驾驶习惯指标.csv"),
    ("stackelberg", "Stackelberg 三场景决策结果", "11_Stackelberg决策结果.csv"),
    ("crossval", "交叉年份验证(2022-23→2024 / 22-24→2025)", "15_交叉年份验证.csv"),
    ("mcconv", "蒙特卡洛收敛性验证", "16_蒙特卡洛收敛图.csv"),
    ("sens", "敏感性分析(龙卷风图数据)", "17_敏感性分析.csv"),
    ("history24", "历史回溯: 2024 摩纳哥红旗案例", "18_历史回溯_2024红旗案例.csv"),
]


def get_tables():
    """按定义顺序返回全部数据表(列名 + 行),供前端"数据总览"渲染。"""
    tables = []
    for key, title, fname in TABLE_SPECS:
        path = os.path.join(CHART_DIR, fname)
        if not os.path.exists(path):
            continue
        rows = _read_csv(path)
        if not rows:
            continue
        cols = list(rows[0].keys())
        tables.append({
            "key": key,
            "title": title,
            "columns": cols,
            "rows": [[str(r.get(c, "")) for c in cols] for r in rows],
        })
    return {"tables": tables}


def get_meta():
    """前端初始化所需的全部静态数据。"""
    return {
        "total_laps": TOTAL_LAPS,
        "drivers": STORE.driver_list(),
        "compounds": [{"code": c, "name": COMPOUND_ZH[c],
                       "color": COMPOUND_COLORS[c]} for c in COMPOUNDS],
        "scenarios": [{"code": k, **v} for k, v in SCENARIOS.items()],
        "tire_template": default_tires(),
        "tire_sources": TIRE_SOURCES,
        "tire_rules": TIRE_RULES,
        "race_sets_max": RACE_SETS_MAX,
        "ai_enabled": bool(_ai_config()[0]),
        "noise": {
            "global_sigma": round(_NOISE_TRIM, 3),
            "mode": "auto",
            "note": "图表/14_圈速波动估计.csv 实测标定(1.5IQR 去尾后近似正态)",
        },
        "defaults": {
            **{k: v for k, v in DEFAULT_PARAMS.items() if k != "compound_offset"},
            "compound_offset": dict(DEFAULT_PARAMS["compound_offset"]),
        },
        "charts": {
            "degradation": "/charts/06_轮胎衰减曲线_C3C4C5.png",
            "drivers_deg": "/charts/06b_车手衰减曲线对比.png",
            "style": "/charts/06c_车手风格分类.png",
            "timeline": "/charts/08_最优停站策略时间线.png",
            "all_drivers": "/charts/08b_车手最优策略对比.png",
            "two_stop": "/charts/08c_车手强制两停策略.png",
            "cumulative": "/charts/09_策略累计时间对比.png",
            "stackelberg": "/charts/11_Stackelberg决策结果.png",
            "critical": "/charts/12_关键拐点.png",
            "tire_choice": "/charts/13_轮胎选择.png",
        },
    }

