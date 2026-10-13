#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
统一的圈次清洗口径(单一事实来源)
=================================

本模块集中定义「哪些圈异常、哪些圈可以进入圈速/衰减拟合」,供
``fit_tire_degradation.py`` / ``fit_lap_noise.py`` /
``plot_lec_monaco_speed.py`` / ``batch_export.py`` 共用,
避免各脚本各写一套判据导致同一年份的标记不一致。

.. important:: 两条链路必须区分

    ① **拟合链路 (fit chain)** —— 圈速回归 / 轮胎衰减曲线 / 圈速波动 σ
       异常圈会污染基准圈速、性能保持率与 σ,**必须剔除**。
       用 ``include_in_fit`` 过滤(见 :func:`fit_ready`)。

    ② **模拟链路 (sim chain)** —— DP / Stackelberg / 蒙特卡洛的 stint 计算
       胎龄是轮胎的真实物理消耗量:红旗、安全车、进站等情形下轮胎照样在磨,
       **这些圈必须照常计入圈数与胎龄**。
       本模块**不删除任何行**,只打标记;模拟侧用 ``TyreAge`` / ``Stint``
       全量累计(``solve_dp_strategy.py`` 的 ``age + 1``、
       ``web\engine.py`` 的 ``lap_profile`` 均逐圈递增)。

统一异常圈判定(自上而下,先命中先归类)
--------------------------------------

=================  ===============================================  ============
原因               判据                                             参与拟合
=================  ===============================================  ============
发车圈             ``LapNumber == 1``                               否
进站圈             ``PitInTime`` 非空(或段末且非末段)                否
中性化圈           ``TrackStatus`` 含 2/4/5/6/7(黄/SC/红旗/VSC)     否
重启圈             前一圈非全绿且紧邻(重启发车程序致圈速偏长)         否
离群圈             同车手+stint 候选圈圈速 MAD > k·1.4826·MAD         否
出场圈             每段第一圈(进站出口)                              是(*)
=================  ===============================================  ============

(*) 出场圈只扣除「进站出口限速」惩罚,保留冷胎暖胎信息参与拟合。

对比旧口径(2024 摩纳哥红旗案例)的差异
--------------------------------------
* 旧 ``fit_tire_degradation`` 用「前圈含 4/5 且本圈含 '1'」判重启,
  且完全漏掉 ``TrackStatus=6/7``(VSC)与 ``26`` 这类黄旗+VSC 圈;
* 旧 ``fit_lap_noise`` 用 ``str.startswith("1")``,会把 2024 摩纳哥
  ``TrackStatus="14"``(SC 重启圈)当成绿旗圈算进 σ;
* 旧 ``plot_lec_monaco_speed`` 用「遥测静止 > 15 s」判 ``interrupted``,
  只有在遥测存在时才可能标出,导致同一年份 ``lap_summary.csv`` 的
  ``是否异常圈`` 因数据可得性而逐车手不一致。

本模块统一为「**全绿才算有效**」(``TrackStatus == "1"``),上述三种写法
一律作废。
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- 常量
GREEN_STATUS = "1"                         # FastF1: 1 = 赛道全清(绿旗)
NEUTRAL_CODES = ("2", "4", "5", "6", "7")  # 黄 / SC / 红旗 / VSC / VSC 结束
NEUTRAL_PATTERN = "[" + "".join(NEUTRAL_CODES) + "]"

DEFAULT_MAD_K = 3.0                       # 离群阈值(倍稳健标准差)
DEFAULT_MAD_MIN_GROUP = 8                 # 组内圈数不足则该组不做离群判定
MIN_OUTLIER_SECONDS = 0.8                 # 离群最小效应量(s):不足此值不改判
FUEL_EFFECT_SEC_PER_LAP = 0.06            # 燃油效应(s/圈),口径统一在此定义

# 排除原因标签(同时用于 lap_summary.csv 的「异常原因」列)
REASON_NONE = ""
REASON_START = "发车圈"
REASON_PIT_IN = "进站圈"
REASON_NEUTRAL = "中性化圈"
REASON_RESTART = "重启圈"
REASON_OUTLIER = "离群圈"

# 供 lap_summary.csv 使用的「是否异常圈」中文值
YES = "是"
NO = "否"

STATUS_SOURCE_FASTF1 = "FastF1.TrackStatus"
STATUS_SOURCE_NONE = "无圈状态数据"


# ----------------------------------------------------------------- 内部工具
def _resolve_group_keys(df, keys):
    """挑出实际存在的分组键;退化为 Driver / 车手。"""
    gk = [k for k in keys if k in df.columns]
    if gk:
        return gk
    for cand in ("Driver", "车手"):
        if cand in df.columns:
            return [cand]
    return []


def _detrend_mad_flags(x, y, mad_k, min_abs=MIN_OUTLIER_SECONDS, degree=None):
    """组内去趋势 → 对残差做 MAD 离群判定。返回布尔数组。

    为什么不能直接对圈速取 MAD
    --------------------------
    单段 stint 的圈速同时叠加了燃油减轻(单调下降)、轮胎衰减与赛道演变,
    曲线是**先快后慢的凸形**(2024 摩纳哥 LEC 一段跑 77 圈,原始圈速从
    84.6 s 降到 78.6 s)。对这样带低频趋势的序列直接取 MAD,会把正常圈
    按趋势的两端判成离群。

    做法
    ----
    1. 按 ``LapNumber`` 拟合 ``degree`` 次多项式(默认:≥15 点用二次,
       否则退化为线性),两轮迭代以削弱离群点对回归线的影响;
    2. 残差减去中位数后再算 MAD → 稳健标准差 ``1.4826·MAD``;
    3. 阈值取 ``max(mad_k·σ_robust, min_abs)`` —— ``min_abs`` 是**最小
       效应量**,保证不足 1 s 的偏差不会被判成异常(对拟合无实质影响)。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if degree is None:
        degree = 2 if n >= 15 else 1
    degree = int(min(degree, max(n - 1, 1)))
    keep = np.ones(n, dtype=bool)
    resid = y - np.median(y)
    for _ in range(2):
        if keep.sum() <= degree:
            break
        try:
            coeffs = np.polyfit(x[keep], y[keep], degree)
        except (np.linalg.LinAlgError, ValueError):
            break
        resid = y - np.polyval(coeffs, x)
        med = np.median(resid[keep])
        mad = np.median(np.abs(resid[keep] - med))
        scale = 1.4826 * mad if mad > 0 else 0.5
        threshold = max(mad_k * scale, min_abs)
        new_keep = np.abs(resid - med) <= threshold
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    scale = 1.4826 * mad if mad > 0 else 0.5
    threshold = max(mad_k * scale, min_abs)
    return np.abs(resid - med) > threshold


# ----------------------------------------------------------------- 主入口
def add_lap_flags(df, mad_k=DEFAULT_MAD_K, mad_min_group=DEFAULT_MAD_MIN_GROUP,
                  mad_col=None, lap_time_col="LapTimeS", fuel_effect=0.0,
                  min_abs=MIN_OUTLIER_SECONDS, group_keys=("Year", "Driver")):
    r"""给圈次表打上统一的清洗标记(**不删行**)。

    参数
    ----
    df : DataFrame
        至少含 ``LapNumber``;若有 ``TrackStatus`` 则启用圈状态判据。
        有 ``Stint`` / ``PitInTime`` / ``PitOutTime`` 时启用进站圈与出场圈判据。
    mad_k, mad_min_group : float, int
        MAD 离群阈值与最小成组圈数。
    mad_col : str or None
        离群检测依据的列。``None``(默认)按优先级自动选取:
        已有的 ``CorrectedLapTime`` → 由 ``lap_time_col`` + ``fuel_effect``
        现算的 ``_mad_basis`` → 原始 ``lap_time_col``。
        **必须用扣除燃油效应的圈速**,否则长 stint 内燃油带来的
        单调下降趋势会让两端正常圈被误判为离群。
    lap_time_col : str
        原始圈速列名(默认 ``LapTimeS``)。
    fuel_effect : float
        燃油效应系数(s/圈),``>0`` 时用于现算 ``_mad_basis``。
    min_abs : float
        离群判定的最小效应量(s),默认 :data:`MIN_OUTLIER_SECONDS`。
    group_keys : tuple
        分组键,默认 ``(Year, Driver)``。

    返回
    ----
    DataFrame
        原表副本,额外包含:

        ``圈状态`` / ``status_source`` / ``is_green`` / ``is_neutralised`` /
        ``is_start_lap`` / ``is_pit_in`` / ``is_pit_out`` / ``is_restart_lap`` /
        ``is_outlier`` / ``is_fit_candidate`` / ``include_in_fit`` /
        ``exclude_reason``。

        其中 ``include_in_fit`` 为 **拟合链路** 的开关,
        ``exclude_reason`` 为被排除的原因(空 = 参与拟合)。
    """
    out = df.copy()
    gk = _resolve_group_keys(out, group_keys)

    # ── ① 圈状态 ─────────────────────────────────────────────
    has_ts = ("TrackStatus" in out.columns and out["TrackStatus"].notna().any())
    if has_ts:
        ts = out["TrackStatus"].astype(str).str.strip()
        out["圈状态"] = ts
        out["is_neutralised"] = ts.str.contains(NEUTRAL_PATTERN, regex=True,
                                                na=False)
        out["is_green"] = ts.eq(GREEN_STATUS)
        out["status_source"] = STATUS_SOURCE_FASTF1
    else:
        # 无圈状态数据(例如仅有遥测 CSV):不做绿旗过滤,交给调用方兜底
        out["圈状态"] = ""
        out["is_neutralised"] = False
        out["is_green"] = True
        out["status_source"] = STATUS_SOURCE_NONE

    # ── ② 发车圈 ─────────────────────────────────────────────
    out["is_start_lap"] = out["LapNumber"] == 1

    # ── ③ 进站圈 / 出场圈 ────────────────────────────────────
    stint_keys = gk + ["Stint"] if ("Stint" in out.columns and gk) else []
    if stint_keys:
        first_of_stint = (out.groupby(stint_keys)["LapNumber"].transform("min")
                          == out["LapNumber"])
        last_of_stint = (out.groupby(stint_keys)["LapNumber"].transform("max")
                         == out["LapNumber"])
        last_stint = (out.groupby(gk)["Stint"].transform("max") == out["Stint"])
        pit_in_from_stint = last_of_stint & ~last_stint
    else:
        first_of_stint = pd.Series(False, index=out.index)
        pit_in_from_stint = pd.Series(False, index=out.index)

    if "PitInTime" in out.columns:
        out["is_pit_in"] = out["PitInTime"].notna() | pit_in_from_stint
    else:
        out["is_pit_in"] = pit_in_from_stint

    if "PitOutTime" in out.columns:
        out["is_pit_out"] = first_of_stint | out["PitOutTime"].notna()
    else:
        out["is_pit_out"] = first_of_stint


    # ── ④ 重启圈:前一圈非全绿(黄/SC/红旗/VSC 后重启发车程序) ──
    out["is_restart_lap"] = False
    if gk and has_ts:
        ordered = out.sort_values(gk + ["LapNumber"])
        gids = ordered.groupby(gk, sort=False).ngroup()
        prev_green = ordered.groupby(gk, sort=False)["is_green"].shift(1)
        prev_lap = ordered.groupby(gk, sort=False)["LapNumber"].shift(1)
        same_group = gids.eq(gids.shift(1))
        contiguous = same_group & prev_lap.notna() & (prev_lap == ordered["LapNumber"] - 1)
        restart = (ordered["is_green"] & prev_green.eq(False) & contiguous)
        restart = restart.reindex(out.index, fill_value=False).astype(bool)
        out.loc[restart, "is_restart_lap"] = True

    # ── ⑤ 拟合候选集(非发车/非进站/全绿/非重启) ─────────────
    cand = ~(out["is_start_lap"].fillna(False)
             | out["is_pit_in"].fillna(False)
             | (~out["is_green"].fillna(True))
             | out["is_restart_lap"].fillna(False))
    out["is_fit_candidate"] = cand

    # ── ⑥ 离群圈:只在候选集内按 车手(+stint) 做 MAD ────────
    # 检测基准必须是"扣除燃油效应后的圈速":长 stint 内圈速随燃油减轻
    # 单调下降(2024 摩纳哥单段可达 8 s)。即便如此,赛道橡胶化带来的
    # 单调趋势仍在,因此再**按 (车手, stint) 线性去趋势**,只对残差做 MAD。
    if mad_col is None:
        if "CorrectedLapTime" in out.columns:
            mad_col = "CorrectedLapTime"
        elif fuel_effect and lap_time_col in out.columns:
            out["_mad_basis"] = (out[lap_time_col]
                                 + fuel_effect * (out["LapNumber"] - 1))
            mad_col = "_mad_basis"
        else:
            mad_col = lap_time_col

    out["is_outlier"] = False
    # 出场圈不参与离群判定:它已由"进站出口惩罚"单独修正,且是暖胎数据来源
    base = cand & ~out["is_pit_out"].fillna(False)
    if mad_col in out.columns and gk and bool(base.any()):
        mad_keys = gk + (["Stint"] if "Stint" in out.columns else [])
        sub = out.loc[base, ["LapNumber", mad_col] + mad_keys]
        sub = sub.dropna(subset=[mad_col])
        for _, part in sub.groupby(mad_keys):
            if len(part) < mad_min_group:
                continue
            idx = part.index.to_numpy()
            x = part["LapNumber"].to_numpy(dtype=float)
            y = part[mad_col].to_numpy(dtype=float)
            flagged = _detrend_mad_flags(x, y, mad_k, min_abs=min_abs)
            if flagged.any():
                out.loc[idx[flagged], "is_outlier"] = True
    out.attrs["cleaning_mad_col"] = mad_col
    out.attrs["cleaning_mad_scope"] = "组内去趋势残差 MAD"
    out.attrs["cleaning_min_abs"] = min_abs

    # ── ⑦ 拟合标记 + 排除原因 ────────────────────────────────
    out["include_in_fit"] = cand & ~out["is_outlier"]

    reason = pd.Series(REASON_NONE, index=out.index, dtype=object)
    reason[out["is_outlier"]] = REASON_OUTLIER
    reason[out["is_restart_lap"]] = REASON_RESTART
    reason[~out["is_green"].fillna(True)] = REASON_NEUTRAL
    reason[out["is_pit_in"].fillna(False)] = REASON_PIT_IN
    reason[out["is_start_lap"].fillna(False)] = REASON_START
    out["exclude_reason"] = reason
    return out


# ----------------------------------------------------------------- 链路助手
def fit_ready(df, drop_pit_out=False):
    """**拟合链路**:取出 ``include_in_fit`` 的子集。

    ``drop_pit_out=True`` 时再剔除出场圈 —— 用于圈速波动 σ 估计:
    出场圈带进站出口限速的结构性偏差,会放大 σ。
    """
    sub = df[df["include_in_fit"]]
    if drop_pit_out and "is_pit_out" in sub.columns:
        sub = sub[~sub["is_pit_out"]]
    return sub


def sim_ready(df):
    """**模拟链路**:原样返回全部圈(含异常圈),胎龄照常累计。

    存在的意义是让调用点显式表达「这里用的是模拟口径」,避免误用
    :func:`fit_ready` 把红旗/SC/进站圈从胎龄序列里删掉。
    """
    return df


def summarise_flags(df, label=""):
    """打印统一的清洗摘要(供各脚本复用),返回统计字典。"""
    n = len(df)
    if not n:
        print(f"  [清洗] {label}无数据")
        return {}
    info = {
        "总圈数": n,
        "参与拟合": int(df["include_in_fit"].sum()),
        "发车圈": int(df["is_start_lap"].sum()),
        "进站圈": int(df["is_pit_in"].sum()),
        "中性化圈": int((~df["is_green"]).sum()),
        "重启圈": int(df["is_restart_lap"].sum()),
        "离群圈": int(df["is_outlier"].sum()),
        "圈状态来源": str(df["status_source"].iloc[0]),
    }
    print(f"  [清洗] {label}总 {info['总圈数']} 圈 → 拟合 {info['参与拟合']} 圈"
          f" | 剔除: 发车 {info['发车圈']} / 进站 {info['进站圈']} / "
          f"中性化 {info['中性化圈']} / 重启 {info['重启圈']} / "
          f"离群 {info['离群圈']}  [{info['圈状态来源']}]")
    return info


def status_export_columns():
    """``*_lap_status.csv`` 列名(供 batch_export.py 与绘图脚本对齐)。"""
    return ["LapNumber", "Driver", "TrackStatus", "Stint", "TyreLife",
            "FreshTyre", "Compound", "PitInTime", "PitOutTime", "LapTime_s"]


def status_csv_path(telemetry_csv):
    """由遥测 CSV 路径推出同目录的圈状态 CSV 路径。

    ``.../2024/LEC/lec_monaco_2024_all_laps_telemetry.csv``
    → ``.../2024/LEC/lec_monaco_2024_lap_status.csv``
    """
    import os
    import re
    folder = os.path.dirname(os.path.abspath(telemetry_csv))
    base = os.path.basename(telemetry_csv)
    m = re.match(r"([a-z]+)_monaco_(\d{4})_", base.lower())
    if m:
        name = f"{m.group(1)}_monaco_{m.group(2)}_lap_status.csv"
    else:
        name = os.path.splitext(base)[0] + "_lap_status.csv"
    return os.path.join(folder, name)

