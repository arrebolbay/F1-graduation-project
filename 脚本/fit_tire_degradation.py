#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
轮胎衰减曲线拟合(C3/C4/C5,剔除C6)

数据源: FastF1 缓存(2022-2025 摩纳哥站正赛 session.laps)
模型:   三阶段衰减(暖胎上升 → 稳定平台 → 衰减下滑)

产出:
    06_轮胎衰减曲线_C3C4C5.png    三配方叠加图
    07_拟合参数表.csv             拟合参数 + R²
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
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- 路径与配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PROJECT_DIR, "缓存", "f1_cache")
CHART_DIR = os.path.join(PROJECT_DIR, "图表")
os.makedirs(CHART_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
import laps_cleaning  # noqa: E402  统一清洗口径(单一事实来源)

YEARS = [2022, 2023, 2024, 2025]
EVENT, SESSION_TYPE = "Monaco", "R"

COMPOUND_MAP = {
    (2022, "SOFT"): "C5", (2022, "MEDIUM"): "C4", (2022, "HARD"): "C3",
    (2023, "SOFT"): "C5", (2023, "MEDIUM"): "C4", (2023, "HARD"): "C3",
    (2024, "SOFT"): "C5", (2024, "MEDIUM"): "C4", (2024, "HARD"): "C3",
    (2025, "SOFT"): "C6", (2025, "MEDIUM"): "C5", (2025, "HARD"): "C4",
}
EXCLUDE_COMPOUNDS = {"C6"}
VALID_COMPOUNDS = ["C5", "C4", "C3"]
COMPOUND_COLORS = {"C5": "#e10600", "C4": "#ffd700", "C3": "#b0b0b0"}

FUEL_EFFECT = laps_cleaning.FUEL_EFFECT_SEC_PER_LAP   # 燃油修正:秒/圈(统一口径)
OUTLIER_MAD_K = laps_cleaning.DEFAULT_MAD_K           # 异常圈剔除阈值(统一口径)

for _n in ("Microsoft YaHei", "SimHei", "SimSun"):
    try:
        fm.findfont(_n, fallback_to_default=False)
    except Exception:
        continue
    plt.rcParams["font.sans-serif"] = [_n, "DejaVu Sans"]
    break
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------- 三阶段模型
def tire_perf(t, P0, t_warm, deg_rate):
    """简化三阶段:暖胎指数上升 → 线性衰减(3参数,更稳定)。

    P(t) = P0 + (100-P0)·(1-exp(-t/t_warm)) - deg_rate·t
    """
    t = np.asarray(t, dtype=float)
    warmup = 100.0 - (100.0 - P0) * np.exp(-t / max(t_warm, 0.1))
    return warmup - deg_rate * t


# ----------------------------------------------------------------- 数据加载
def load_all_laps():
    import fastf1
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    frames = []
    for year in YEARS:
        print(f"  加载 {year} 年 ...", end=" ", flush=True)
        try:
            sess = fastf1.get_session(year, EVENT, SESSION_TYPE)
            sess.load(telemetry=False, weather=False, messages=False)
        except Exception as exc:
            print(f"失败: {exc}")
            continue
        laps = sess.laps.copy()
        laps["Year"] = year
        frames.append(laps)
        print(f"OK({len(laps)} 圈)")
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------- 数据清洗
def build_clean_table(raw: pd.DataFrame):
    """清洗 + 打统一标记。返回 ``(df_fit, df_all)``。

    * ``df_fit`` —— **拟合链路**:``include_in_fit`` 子集,用于衰减拟合;
    * ``df_all`` —— **模拟链路**:全量(含异常圈),``TyreAge`` 连续累计。

    异常圈判定统一走 :mod:`laps_cleaning`,本函数不再自带一套规则。
    """
    df = raw.copy()
    df = df.dropna(subset=["LapTime", "Compound", "LapNumber", "Driver"])
    df["LapTimeS"] = df["LapTime"].dt.total_seconds()

    df["AbsCompound"] = [
        COMPOUND_MAP.get((y, c), "UNKNOWN")
        for y, c in zip(df["Year"], df["Compound"])
    ]
    n = len(df)
    df = df[~df["AbsCompound"].isin(EXCLUDE_COMPOUNDS | {"UNKNOWN"})]
    print(f"  配方过滤: {n} → {len(df)} 行")

    df = df.sort_values(["Year", "Driver", "Stint", "LapNumber"])

    # 胎龄计算:优先用 FastF1 的 TyreLife(含旧胎历史),否则用 cumcount
    if "TyreLife" in df.columns and df["TyreLife"].notna().any():
        df["TyreAge"] = df["TyreLife"].fillna(1) - 1   # TyreLife 从1开始,转为0起
    else:
        df["TyreAge"] = df.groupby(["Year", "Driver", "Stint"]).cumcount()

    # 旧胎修正:起步用的排位赛旧胎 / 换胎时的旧胎,胎龄起点 > 0
    # 思路B:偏移量由 fit_one_compound 中的基准曲线反推,这里不再用固定 +2
    if "FreshTyre" in df.columns:
        first_stint = df["Stint"] == df.groupby(["Year", "Driver"])["Stint"].transform("min")
        used_first = first_stint & (~df["FreshTyre"].astype(bool))
        n_used = int(used_first.sum())
        if n_used:
            print(f"  旧胎识别: 首段 {n_used} 圈使用旧胎(偏移量由基准曲线反推)")

    # 注意:include_in_fit 由 laps_cleaning.add_lap_flags 统一判定(见下方)

    # ① 每段第一圈 = 出场圈 → 估计并扣除进站出口惩罚后参与拟合(暖胎数据!)
    first_lap = df.groupby(["Year", "Driver", "Stint"])["LapNumber"].transform("min") == df["LapNumber"]
    df["is_pit_out"] = first_lap

    # 估计进站出口惩罚:出场圈 vs 第二圈的中位数差(含进站出口限速+冷胎效应)
    penalties = []
    for _, grp in df.groupby(["Year", "Driver", "Stint"]):
        if len(grp) >= 2 and grp["is_pit_out"].iloc[0]:
            grp = grp.sort_values("TyreAge")
            penalties.append(grp.iloc[0]["LapTimeS"] - grp.iloc[1]["LapTimeS"])
    pit_exit_penalty = float(np.median(penalties)) if penalties else 15.0
    print(f"  进站出口惩罚估计: {pit_exit_penalty:.1f} 秒")

    # 修正出场圈圈速(减去进站出口限速惩罚,保留冷胎效应)
    df.loc[df["is_pit_out"], "LapTimeS"] -= pit_exit_penalty

    # ── 燃油修正(统一口径常量 laps_cleaning.FUEL_EFFECT_SEC_PER_LAP) ──
    df["CorrectedLapTime"] = df["LapTimeS"] + FUEL_EFFECT * (df["LapNumber"] - 1)

    # ── 统一异常圈判定(单一事实来源:laps_cleaning.add_lap_flags) ──
    #    发车圈 / 进站圈 / 中性化圈(黄·SC·红旗·VSC) / 重启圈 / 离群圈
    #    旧实现的三处不一致(只认 [45]、startswith("1") 漏判 VSC、
    #    对未扣燃油的原始圈速取 MAD)已全部作废。
    df = laps_cleaning.add_lap_flags(df, mad_k=OUTLIER_MAD_K,
                                     mad_col="CorrectedLapTime")
    laps_cleaning.summarise_flags(df, "全站(2022-2025 摩纳哥) ")

    # 性能保持率:按 (车手, 配方) 归一化,消除车手/赛车差异
    best = df.groupby(["Year", "Driver", "AbsCompound"])["CorrectedLapTime"].transform("min")
    df["PerfPct"] = best / df["CorrectedLapTime"] * 100.0

    # 链路分离:
    #   ① 拟合链路 = include_in_fit 子集(剔除发车/进站/中性化/重启/离群圈)
    #   ② 模拟链路 = 全量,胎龄 TyreAge 连续累计(供 DP/MC 口径核对)
    df_fit = laps_cleaning.fit_ready(df).reset_index(drop=True)
    df_all = laps_cleaning.sim_ready(df).reset_index(drop=True)
    return df_fit, df_all


# ----------------------------------------------------------------- 拟合
INIT_GUESS = {
    "C5": [96.0, 2.0, 0.03],
    "C4": [97.0, 2.5, 0.02],
    "C3": [97.5, 3.0, 0.01],
}
BOUNDS = ([85, 0.2, 0.0], [100, 20, 0.2])


def fit_one_compound(df_comp, compound, n_boot=0, seed=42, min_count=3):
    """思路B:先用全新胎拟合基准曲线,再反推旧胎的等效胎龄,最后统一拟合。

    ``n_boot > 0`` 时额外做 stint 级自助重采样,给出 deg_rate 的 95% 置信区间
    (只在全场汇总调用;车手级 90 次拟合并行跑 bootstrap 太慢)。
    ``min_count`` 为每个胎龄点所需的最少圈数: 全场汇总用 3(稳健),
    车手级用 2(否则 50 圈样本的车手也会被判"拟合失败"而整段回退全场参数)。
    """
    binned = df_comp.groupby("TyreAge")["PerfPct"].agg(
        ["median", "mean", "std", "count"]).reset_index()
    binned = binned[binned["count"] >= min_count]
    t_data = binned["TyreAge"].to_numpy(dtype=float)
    p_data = binned["median"].to_numpy(dtype=float)
    p_std = binned["std"].fillna(0).to_numpy(dtype=float)
    if len(t_data) < 3:
        raise ValueError(f"样本不足({len(t_data)} 个胎龄点)")
    # 第1步:用全新胎(FreshTyre=True)拟合基准曲线
    if "FreshTyre" in df_comp.columns:
        new_mask = df_comp["FreshTyre"].astype(bool)
        df_new = df_comp[new_mask]
        df_used = df_comp[~new_mask]
    else:
        df_new = df_comp
        df_used = pd.DataFrame()

    # 基准曲线:用全新胎的分箱中位数 + 线性插值
    if len(df_new) >= 20:
        base_bin = df_new.groupby("TyreAge")["PerfPct"].median().dropna()
        base_t = base_bin.index.to_numpy(dtype=float)
        base_p = base_bin.to_numpy(dtype=float)
        # 排序后插值
        sort_idx = np.argsort(base_t)
        base_t, base_p = base_t[sort_idx], base_p[sort_idx]

        # 第2步:对旧胎圈,反推等效胎龄
        if len(df_used) >= 5:
            # 基准曲线的衰减段斜率(用于线性外推)
            if len(base_t) >= 5:
                slope, _ = np.polyfit(base_t[base_t > 2], base_p[base_t > 2], 1)
            else:
                slope = -0.05
            if slope >= -1e-6:
                slope = -0.05  # 保底

            # 旧胎的性能差 → 等效胎龄偏移
            used_ages = df_used["TyreAge"].to_numpy(dtype=float)
            used_perf = df_used["PerfPct"].to_numpy(dtype=float)
            # 在基准曲线上插值出该胎龄的预期性能
            expected = np.interp(used_ages, base_t, base_p,
                                 left=base_p[0], right=base_p[-1])
            deficit = expected - used_perf  # 正数 = 比预期慢
            offset = np.where(deficit > 0, deficit / abs(slope), 0.0)
            offset = np.clip(offset, 0, 15)  # 上限15圈
            # 更新胎龄
            df_comp = df_comp.copy()
            df_comp.loc[~new_mask, "TyreAge"] = used_ages + offset
            print(f"    [{compound}] 旧胎修正: {len(df_used)} 圈, "
                  f"平均偏移 {offset.mean():.1f} 圈")

    # ── 重新分箱(用修正后的胎龄) ──
    binned = df_comp.groupby("TyreAge")["PerfPct"].agg(
        ["median", "mean", "std", "count"]).reset_index()
    binned = binned[binned["count"] >= min_count]
    t_data = binned["TyreAge"].to_numpy(dtype=float)
    p_data = binned["median"].to_numpy(dtype=float)
    p_std = binned["std"].fillna(0).to_numpy(dtype=float)

    # 移动平均趋势(窗口=5)
    window = min(5, len(p_data))
    if window >= 3:
        trend = pd.Series(p_data).rolling(window, center=True, min_periods=1).median().to_numpy()
    else:
        trend = p_data.copy()

    # 关键统计量
    warm_mask = t_data <= 2
    P0 = float(p_data[warm_mask].min()) if warm_mask.any() else float(p_data[0])
    peak_idx = int(np.argmax(trend))
    t_peak = float(t_data[peak_idx])
    peak_val = float(trend[peak_idx])

    # 衰减率:胎龄>5 的**加权**线性回归斜率(权重 = 分箱样本数)
    weights = binned["count"].to_numpy(dtype=float)
    deg_mask = t_data > 5
    if deg_mask.sum() >= 5:
        deg_rate = -weighted_slope(t_data[deg_mask], p_data[deg_mask],
                                   weights[deg_mask])
        deg_se = weighted_slope_se(t_data[deg_mask], p_data[deg_mask],
                                   weights[deg_mask])
    else:
        deg_rate = 0.0
        deg_se = float("inf")

    params = {"P0": P0, "t_peak": t_peak, "peak": peak_val, "deg": deg_rate,
              "peak_val": peak_val, "deg_rate": deg_rate}
    stats = model_fit_stats(t_data, p_data, weights, params)
    r2, rmse, r2_deg = stats["r2_all"], stats["rmse"], stats["r2_deg"]
    ci_low, ci_high, n_boot_ok = bootstrap_deg_ci(df_comp, n_boot, seed)
    n_stints = int(df_comp.groupby(["Year", "Driver", "Stint"]).ngroups) \
        if "Stint" in df_comp.columns else 0

    print(f"    [{compound}] 样本{int(binned['count'].sum())}  "
          f"P0={P0:.1f}%  峰值={peak_val:.1f}%@{t_peak:.0f}圈  "
          f"衰减={deg_rate:.3f}%/圈  范围={p_data.min():.1f}~{p_data.max():.1f}%"
          + (f"  R²(衰减段)={r2_deg:.3f} RMSE={rmse:.2f}"
             if np.isfinite(r2_deg) else "")
          + (f"  95%CI=[{ci_low:.3f},{ci_high:.3f}]"
             if np.isfinite(ci_low) else ""))

    return {
        "P0": P0, "t_peak": t_peak, "peak_val": peak_val,
        "deg_rate": deg_rate,
        "t": t_data, "p": p_data, "trend": trend, "std": p_std,
        "n": int(binned["count"].sum()),
        "r2": r2, "r2_deg": r2_deg, "rmse": rmse,
        "deg_ci_low": ci_low, "deg_ci_high": ci_high,
        "deg_se": deg_se,
        "n_boot": n_boot_ok, "n_stints": n_stints,
    }


def perf_pct_from_params(t, params):
    """从拟合参数计算性能保持率(%)——供 DP 使用。"""
    P0, peak_val = params["P0"], params["peak_val"]
    t_peak, deg_rate = params["t_peak"], params["deg_rate"]
    if t <= t_peak:
        return P0 + (peak_val - P0) * (t / max(t_peak, 1))
    return peak_val - deg_rate * (t - t_peak)


# ----------------------------------------------------------------- 不确定性量化
def weighted_slope(t, y, w):
    """加权最小二乘斜率(截距自由)。

    旧版用 ``np.polyfit`` 等权回归:胎龄 3 圈的分箱和 200 圈的分箱权重相同,
    长胎龄段(也是唯一能识别 deg 的段)反而被样本稀疏的分箱拉偏。
    这里以分箱样本数为权重 —— 与"每个胎龄点的可靠性"一致。
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = w.sum()
    if sw <= 0 or len(t) < 2:
        return 0.0
    t_bar = (w * t).sum() / sw
    y_bar = (w * y).sum() / sw
    denom = (w * (t - t_bar) ** 2).sum()
    if denom <= 1e-12:
        return 0.0
    return float((w * (t - t_bar) * (y - y_bar)).sum() / denom)


def weighted_slope_se(t, y, w):
    """加权最小二乘斜率的标准误(用于经验贝叶斯收缩)。

    SE² = σ̂² / Σ w(t−t̄)²,其中 σ̂² = Σ w·r² / (Σw − 2)。
    胎龄点只有 2~3 个的稀疏拟合会得到极大的 SE —— 正是我们要"自动降权"的对象。
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = w.sum()
    if len(t) < 4 or sw <= 0:
        return float("inf")
    t_bar = (w * t).sum() / sw
    sxx = float((w * (t - t_bar) ** 2).sum())
    if sxx <= 1e-12:
        return float("inf")
    slope = weighted_slope(t, y, w)
    resid = y - (y.mean() + slope * (t - t.mean()))
    dof = max(sw - 2.0, 1.0)
    sigma2 = float((w * resid ** 2).sum() / dof)
    se = float(np.sqrt(max(sigma2, 0.0) / sxx))
    return se if np.isfinite(se) else float("inf")


def model_fit_stats(t, y, w, params, t_deg=5.0):
    """三阶段模型对分箱中位数的拟合优度。

    返回 ``{"r2_all", "rmse", "r2_deg"}``:

    * ``r2_all`` —— 全胎龄段的加权 R²(含暖胎段);
    * ``r2_deg`` —— **衰减段(t > t_deg)的加权 R²**,即 deg_rate 真正生效的区间
      (C3 的"暖胎段"其实是燃油/赛道演化的残差趋势,全段 R² 会被它拉负,
      单看全段会误判"模型没用",故分开报告);
    * ``rmse`` —— 加权残差标准差(单位:性能点)。
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    nan = float("nan")
    if len(t) < 2:
        return {"r2_all": nan, "rmse": nan, "r2_deg": nan}

    def _r2(mask):
        sw = w[mask].sum()
        if sw <= 0 or int(mask.sum()) < 2:
            return nan
        yhat = np.array([perf_pct_from_params(ti, params) for ti in t[mask]])
        resid = y[mask] - yhat
        ss_res = float((w[mask] * resid ** 2).sum())
        y_bar = float((w[mask] * y[mask]).sum() / sw)
        ss_tot = float((w[mask] * (y[mask] - y_bar) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else nan

    yhat = np.array([perf_pct_from_params(ti, params) for ti in t])
    sw = float(w.sum())
    rmse = float(np.sqrt((w * (y - yhat) ** 2).sum() / sw)) if sw > 0 else nan
    return {"r2_all": _r2(np.ones_like(t, dtype=bool)),
            "rmse": rmse,
            "r2_deg": _r2(t > t_deg)}


def bootstrap_deg_ci(df_comp, n_boot=500, seed=42, min_count=3, t_min=5):
    """deg_rate 的 95% 置信区间(**stint 级**自助重采样)。

    同一 stint 内的圈高度相关(轮胎是同一条、赛道状态连续),按"圈"重采样会
    严重低估不确定性;按 (年份, 车手, stint) 整段重采样才是正确的聚类口径。
    返回 ``(下界, 上界, 有效重采样次数)``,样本不足时返回 ``(nan, nan, 0)``。
    """
    if n_boot <= 0 or "Stint" not in df_comp.columns:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df_comp.groupby(["Year", "Driver", "Stint"])]
    if len(groups) < 5:
        return float("nan"), float("nan"), 0
    vals = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(groups), len(groups))
        d = pd.concat([groups[i] for i in pick], ignore_index=True)
        b = d.groupby("TyreAge")["PerfPct"].agg(["median", "count"]).reset_index()
        b = b[b["count"] >= min_count]
        m = b["TyreAge"] > t_min
        if int(m.sum()) < 4:
            continue
        slopes = -weighted_slope(b.loc[m, "TyreAge"], b.loc[m, "median"],
                                 b.loc[m, "count"])
        vals.append(slopes)
    if len(vals) < 20:
        return float("nan"), float("nan"), len(vals)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)),
            len(vals))


def shrink_to_pooled(res, pooled, n, k=30, min_n=15, ci=None):
    """经验贝叶斯收缩: 车手级参数按 ``w = n_eff/(n_eff+k)`` 向全场汇总收缩,并截断到全场 CI。

    为什么要收缩: 07b 里出现 deg ≤ 0(轮胎越跑越快,如 BOT C3 −0.1157)与
    "样本 6 圈却当独立结论"的行 —— 单个车手-配方的样本量(常 6~100 圈)不足以
    支撑 4 个参数,直接丢进 DP 会解出 C4(73圈)→C5(5圈) 这类贴着约束边界的伪最优。

    三处约束:
      1. ``n_eff`` 用**参与拟合的分箱样本数**(``res["n"]``)而不是原始行数;
         ``n_eff < min_n`` 时 w=0(完全采用全场参数)。
      2. ``w = n_eff/(n_eff+k)``,k=30 圈 ≈ "30 圈的全场先验";100 圈样本 → w=0.77。
      3. 混合后 ``deg`` 截断到 ``[0, ci_high]``(ci 为全场 deg 的 95% 置信区间):
         超出数据所能支撑的范围就按上界截断 —— 否则 44 圈的稀疏拟合会给出
         18%/百圈这种把 DP 解推向约束边界的伪信号。

    注: 也尝试过用斜率标准误做"完全经验贝叶斯"(w=τ²/(τ²+SE²)),但 2~3 个胎龄点
    的稀疏拟合会把残差方差低估、τ² 被放大到超过全场水平,反而失去收缩作用,
    故改用上面这套参数更少、可复现的口径。
    """
    res = dict(res)
    n_eff = int(res.get("n", n) or 0)
    w = 0.0 if n_eff < min_n else float(n_eff) / (float(n_eff) + float(k))
    for key in ("P0", "peak_val", "t_peak", "deg_rate"):
        res[key] = w * float(res[key]) + (1.0 - w) * float(pooled[key])

    hi = float("inf")
    if ci is not None and len(ci) == 2 and np.isfinite(ci[1]):
        hi = float(ci[1])
    res["deg_rate"] = min(max(float(res["deg_rate"]), 0.0), hi)
    res["shrink_w"] = round(w, 3)
    res["n"] = n_eff
    if n_eff < min_n:
        res["source"] = "pooled(样本不足)"
    elif res.get("source") == "pooled":
        res["source"] = "pooled(拟合失败)"
    else:
        res["source"] = f"driver(收缩{w:.2f})"
    return res



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


def fit_per_driver(df, pooled_results, k=30, min_n=15):
    """按车手拟合衰减曲线,并向全场汇总做经验贝叶斯收缩。

    流程: 单车手-配方拟合(样本<3 个胎龄点或失败 → 直接用全场参数)
      → ``shrink_to_pooled``: w = n_eff/(n_eff+k) 混合车手估计与全场估计,
        并把 deg 截断到全场 95% CI 内、非负。

    收缩强度 k=30 圈: 40 圈样本 → w=0.57,100 圈 → w=0.77,328 圈 → w=0.92。
    """
    driver_results = {}
    for driver in sorted(df["Driver"].unique()):
        driver_results[driver] = {}
        for comp in VALID_COMPOUNDS:
            sub = df[(df["Driver"] == driver) & (df["AbsCompound"] == comp)]
            n = len(sub)
            try:
                result = fit_one_compound(sub, comp, min_count=2)
                result["source"] = "driver"
            except (IndexError, ValueError):
                result = dict(pooled_results[comp])
                result["source"] = "pooled"
                result["n"] = n
            pooled = pooled_results[comp]
            driver_results[driver][comp] = shrink_to_pooled(
                result, pooled, n, k=k, min_n=min_n,
                ci=(pooled.get("deg_ci_low"), pooled.get("deg_ci_high")))
    return driver_results


# ----------------------------------------------------------------- 绘图
def plot_curves(fit_results, df, out_path):
    fig, ax = plt.subplots(figsize=(12, 7))
    for comp in VALID_COMPOUNDS:
        sub = df[df["AbsCompound"] == comp]
        ax.scatter(sub["TyreAge"], sub["PerfPct"], s=6, alpha=0.12,
                   color=COMPOUND_COLORS[comp], edgecolors="none")

    t_fine = np.linspace(0, 55, 300)
    for comp in VALID_COMPOUNDS:
        r = fit_results[comp]
        label = (f"{comp}  P₀={r['P0']:.1f}%  峰值@{r['t_peak']:.0f}圈  "
                 f"衰减={r['deg_rate']*100:.2f}%/百圈")
        ax.plot(r["t"], r["trend"], lw=2.5,
                color=COMPOUND_COLORS[comp], label=label, zorder=5)
        ax.fill_between(r["t"], r["trend"] - r["std"], r["trend"] + r["std"],
                        alpha=0.2, color=COMPOUND_COLORS[comp])

    ax.axhline(100, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("胎龄(圈)", fontsize=12)
    ax.set_ylabel("性能保持率 (%)", fontsize=12)
    ax.set_title("C3 / C4 / C5 轮胎衰减曲线(摩纳哥 2022-2025)", fontsize=14)
    ax.set_xlim(-1, 56)
    ax.set_ylim(85, 101)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


def save_params(fit_results, out_path):
    """全场汇总参数表: 4 个参数 + 拟合优度 + deg 的 95% 置信区间。

    旧版文件头写着"拟合参数 + R²",实际并没有输出 R²(纯文档与代码不一致);
    现在 R²/RMSE/CI 都是真算出来的,敏感性分析可以直接引用 CI 而不是拍 ±50%。
    """
    rows = []
    for comp in VALID_COMPOUNDS:
        r = fit_results[comp]
        rows.append({
            "配方": comp,
            "P0_pct": round(r["P0"], 2),
            "t_peak_laps": round(r["t_peak"], 1),
            "peak_pct": round(r["peak_val"], 2),
            "deg_rate_pct_per_lap": round(r["deg_rate"], 4),
            "deg_rate_pct_per_100laps": round(r["deg_rate"] * 100, 2),
            "样本数": r["n"],
            "stint数": r.get("n_stints", 0),
            "deg_ci95_low": round(r["deg_ci_low"], 4) if np.isfinite(r.get("deg_ci_low", np.nan)) else "",
            "deg_ci95_high": round(r["deg_ci_high"], 4) if np.isfinite(r.get("deg_ci_high", np.nan)) else "",
            "R2": round(r["r2"], 4) if np.isfinite(r.get("r2", np.nan)) else "",
            "R2_衰减段": round(r["r2_deg"], 4) if np.isfinite(r.get("r2_deg", np.nan)) else "",
            "RMSE_性能点": round(r["rmse"], 4) if np.isfinite(r.get("rmse", np.nan)) else "",
            "bootstrap次数": r.get("n_boot", 0),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {os.path.basename(out_path)}")


def plot_driver_curves(driver_results, out_path):
    """按车手衰减曲线对比(1×3 子图,颜色=衰减率)。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    t_fine = np.linspace(0, 50, 100)
    for idx, comp in enumerate(VALID_COMPOUNDS):
        ax = axes[idx]
        for driver in sorted(driver_results):
            r = driver_results[driver][comp]
            p_fine = [perf_pct_from_params(t, r) for t in t_fine]
            norm = min(r["deg_rate"] / 0.15, 1.0)
            color = plt.cm.RdYlGn_r(norm)
            lbl = f"{driver}({r['deg_rate']*100:.1f})"
            ax.plot(t_fine, p_fine, lw=1.2, color=color, alpha=0.7, label=lbl)
        ax.set_title(f"{comp} 衰减曲线(按车手)", fontsize=12)
        ax.set_xlabel("胎龄(圈)")
        if idx == 0:
            ax.set_ylabel("性能保持率 (%)")
        ax.legend(fontsize=5.5, ncol=2, loc="lower left")
        ax.grid(alpha=0.3, ls=":")
    fig.suptitle("各车手轮胎衰减曲线对比(颜色=衰减率:红快→绿慢)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


def save_driver_params(driver_results, out_path):
    """按车手衰减参数表。"""
    rows = []
    for driver in sorted(driver_results):
        for comp in VALID_COMPOUNDS:
            r = driver_results[driver][comp]
            rows.append({
                "车手": driver,
                "中文名": DRIVER_NAMES.get(driver, driver),
                "配方": comp,
                "P0_pct": round(r["P0"], 2),
                "t_peak_laps": round(r["t_peak"], 1),
                "peak_pct": round(r["peak_val"], 2),
                "deg_rate_pct_per_lap": round(r["deg_rate"], 4),
                "样本数": r["n"],
                "收缩权重": r.get("shrink_w", ""),
                "来源": r.get("source", "driver"),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {os.path.basename(out_path)}")


def compute_driver_metrics(driver_results, df):
    """计算每位车手的驾驶习惯/轮胎消耗/燃油消耗/保胎策略指标。"""
    FUEL_PER_KG = 0.035      # 燃油效应:秒/公斤
    fuel_per_lap = FUEL_EFFECT / FUEL_PER_KG   # kg/lap
    metrics = {}
    for driver in sorted(driver_results):
        results = driver_results[driver]
        deg_rates = [results[c]["deg_rate"] for c in VALID_COMPOUNDS]
        avg_deg = float(np.mean(deg_rates))
        conservation = max(0.0, 100.0 - avg_deg * 800)
        if avg_deg < 0.03:
            style = "保胎型"
        elif avg_deg < 0.08:
            style = "均衡型"
        else:
            style = "激进型"
        # 基准圈速:该车手的最短修正圈速(反映赛车+车手综合性能)
        drv_df = df[df["Driver"] == driver]
        base_time = float(drv_df["CorrectedLapTime"].min()) if len(drv_df) > 0 else 75.0
        metrics[driver] = {
            "avg_deg_rate": avg_deg,
            "fuel_per_lap": fuel_per_lap,
            "conservation": conservation,
            "style": style,
            "base_time": base_time,
        }
    return metrics


def save_driver_metrics(metrics, out_path):
    """保存车手驾驶习惯/消耗指标表。"""
    rows = []
    for driver in sorted(metrics):
        m = metrics[driver]
        rows.append({
            "车手": driver,
            "中文名": DRIVER_NAMES.get(driver, driver),
            "平均衰减率_pct_per_lap": round(m["avg_deg_rate"], 4),
            "燃油消耗_kg_per_lap": round(m["fuel_per_lap"], 2),
            "保胎评分": round(m["conservation"], 1),
            "驾驶风格": m["style"],
            "基准圈速_s": round(m["base_time"], 2),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {os.path.basename(out_path)}")


def plot_driver_classification(metrics, out_path):
    """车手驾驶风格分类散点图(横轴=衰减率,纵轴=保胎评分)。"""
    fig, ax = plt.subplots(figsize=(12, 8))
    style_colors = {"保胎型": "#2ca25f", "均衡型": "#ffd700", "激进型": "#e10600"}
    for driver in sorted(metrics):
        m = metrics[driver]
        color = style_colors.get(m["style"], "#888")
        ax.scatter(m["avg_deg_rate"] * 100, m["conservation"],
                   s=120, color=color, edgecolors="black", linewidth=0.5, zorder=5)
        ax.annotate(DRIVER_NAMES.get(driver, driver),
                    (m["avg_deg_rate"] * 100, m["conservation"]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")
    ax.axvline(3, color="gray", ls=":", lw=0.8)
    ax.axvline(8, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("平均轮胎衰减率 (%/百圈)", fontsize=12)
    ax.set_ylabel("保胎评分", fontsize=12)
    ax.set_title("车手驾驶风格分类(颜色=风格类型)", fontsize=14)
    # 图例
    for style, color in style_colors.items():
        ax.scatter([], [], s=80, color=color, label=style, edgecolors="black")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


# ----------------------------------------------------------------- 主流程
def main():
    print("=" * 55)
    print(" 轮胎衰减曲线拟合 (C3 / C4 / C5)")
    print("=" * 55)

    print("\n[1] 加载数据 ...")
    raw = load_all_laps()

    print("\n[2] 清洗数据 ...")
    df, df_all = build_clean_table(raw)
    print(f"  拟合链路(参与拟合): {len(df)} 行")
    print(f"  模拟链路(全量含异常圈): {len(df_all)} 行 —— 胎龄 TyreAge 连续")
    print(f"  各配方样本: {df['AbsCompound'].value_counts().to_dict()}")

    # 圈次清洗标记明细(便于核对"剔除的圈仍计入胎龄"这一口径)
    keep_cols = ["Year", "Driver", "LapNumber", "Stint", "TyreAge",
                 "AbsCompound", "圈状态", "LapTimeS", "CorrectedLapTime",
                 "is_start_lap", "is_pit_in", "is_pit_out", "is_neutralised",
                 "is_restart_lap", "is_outlier", "include_in_fit",
                 "exclude_reason"]
    detail_path = os.path.join(CHART_DIR, "07d_圈次清洗标记明细.csv")
    df_all[[c for c in keep_cols if c in df_all.columns]].to_csv(
        detail_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {os.path.basename(detail_path)}")

    print("\n[3] 拟合三阶段模型(思路B+暖胎恢复:出场圈修正后参与拟合) ...")
    fit_results = {}
    n_boot = int(os.getenv("F1_DEG_BOOT", "500"))
    for comp in VALID_COMPOUNDS:
        sub = df[df["AbsCompound"] == comp]
        if len(sub) < 10:
            print(f"    [{comp}] 样本不足({len(sub)}),跳过")
            continue
        fit_results[comp] = fit_one_compound(sub, comp, n_boot=n_boot)

    print("\n[4] 按车手拟合衰减曲线 ...")
    driver_results = fit_per_driver(df, fit_results)
    plot_driver_curves(driver_results,
                       os.path.join(CHART_DIR, "06b_车手衰减曲线对比.png"))
    save_driver_params(driver_results,
                       os.path.join(CHART_DIR, "07b_车手衰减参数表.csv"))

    print("\n[5] 车手驾驶习惯/消耗指标 ...")
    metrics = compute_driver_metrics(driver_results, df)
    save_driver_metrics(metrics,
                        os.path.join(CHART_DIR, "07c_车手驾驶习惯指标.csv"))
    plot_driver_classification(metrics,
                               os.path.join(CHART_DIR, "06c_车手风格分类.png"))

    print("\n[6] 输出 ...")
    plot_curves(fit_results, df,
                os.path.join(CHART_DIR, "06_轮胎衰减曲线_C3C4C5.png"))
    save_params(fit_results,
                os.path.join(CHART_DIR, "07_拟合参数表.csv"))
    print("\n完成!(旧版结果已保存为 06_轮胎衰减曲线_v1旧版.png)")


if __name__ == "__main__":
    main()

