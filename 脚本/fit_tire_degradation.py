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

FUEL_EFFECT = 0.06               # 燃油修正:秒/圈(可调)
OUTLIER_MAD_K = 3.0              # 异常圈剔除阈值

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
def build_clean_table(raw: pd.DataFrame) -> pd.DataFrame:
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

    # 标记 include_in_fit:发车圈/进站圈 不参与拟合,但胎龄照常累计
    df["include_in_fit"] = True

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

    # ② 每段最后一圈(非末段)= 进站圈(含进站损失时间,圈速人为偏长)
    df["_last_stint"] = df.groupby(["Year", "Driver"])["Stint"].transform("max") == df["Stint"]
    df["_last_lap"] = df.groupby(["Year", "Driver", "Stint"])["LapNumber"].transform("max") == df["LapNumber"]
    pit_in = df["_last_lap"] & ~df["_last_stint"]
    df.loc[pit_in, "include_in_fit"] = False

    # ③ 第1圈 = 发车圈(发车程序导致圈速偏长)
    df.loc[df["LapNumber"] == 1, "include_in_fit"] = False

    # ④ 红旗/安全车重启后的第一圈(重启发车程序导致圈速偏长)
    if "TrackStatus" in df.columns:
        def _flag_restart(g):
            g = g.sort_values("LapNumber").copy()
            prev_ts = g["TrackStatus"].astype(str).shift(1)
            curr_ts = g["TrackStatus"].astype(str)
            # 前一圈有红旗(5)或SC(4),当前圈绿旗(1) → 重启圈
            is_restart = (
                prev_ts.str.contains("[45]", regex=True, na=False) &
                curr_ts.str.contains("1", na=False)
            )
            g.loc[is_restart, "include_in_fit"] = False
            return g

        df = df.groupby(["Year", "Driver"], group_keys=False).apply(_flag_restart)

    n_excl = int((~df["include_in_fit"]).sum())
    print(f"  标记排除圈: {n_excl} 行(发车/进站/重启,出场圈已修正后参与拟合)")

    # 诊断:2024年 Lap2 数据检查
    diag = df[(df["Year"] == 2024) & (df["LapNumber"] == 2)]
    if not diag.empty:
        print(f"  [诊断] 2024 Lap2: {len(diag)} 行, "
              f"排除 {int((~diag['include_in_fit']).sum())} 行")
        for drv in ["LEC", "VER", "HAM"]:
            row = diag[diag["Driver"] == drv]
            if not row.empty:
                r = row.iloc[0]
                ft = r.get("FreshTyre", "N/A")
                print(f"    {drv}: LapTime={r['LapTimeS']:.1f}s  "
                      f"TyreAge={r['TyreAge']:.0f}  {r['AbsCompound']}  "
                      f"FreshTyre={ft}  拟合={r['include_in_fit']}")

    # ④ 异常圈过滤(SC/VSC/交通,MAD)—— 也标记为不参与拟合
    def _flag_outliers(g):
        med = g["LapTimeS"].median()
        mad = (g["LapTimeS"] - med).abs().median()
        if mad < 1e-6:
            mad = 0.5
        outlier = (g["LapTimeS"] - med).abs() > OUTLIER_MAD_K * 1.4826 * mad
        g = g.copy()
        g.loc[outlier, "include_in_fit"] = False
        return g

    df = df.groupby(["Year", "Driver"], group_keys=False).apply(_flag_outliers)
    n_excl2 = int((~df["include_in_fit"]).sum())
    print(f"  加上异常圈后排除: {n_excl2} 行")
    print(f"  参与拟合的有效行: {int(df['include_in_fit'].sum())} 行")

    # 燃油修正
    df["CorrectedLapTime"] = df["LapTimeS"] + FUEL_EFFECT * (df["LapNumber"] - 1)

    # 性能保持率:按 (车手, 配方) 归一化,消除车手/赛车差异
    best = df.groupby(["Year", "Driver", "AbsCompound"])["CorrectedLapTime"].transform("min")
    df["PerfPct"] = best / df["CorrectedLapTime"] * 100.0

    # 只保留参与拟合的行
    df_fit = df[df["include_in_fit"]].copy()
    return df_fit.reset_index(drop=True)


# ----------------------------------------------------------------- 拟合
INIT_GUESS = {
    "C5": [96.0, 2.0, 0.03],
    "C4": [97.0, 2.5, 0.02],
    "C3": [97.5, 3.0, 0.01],
}
BOUNDS = ([85, 0.2, 0.0], [100, 20, 0.2])


def fit_one_compound(df_comp, compound):
    """思路B:先用全新胎拟合基准曲线,再反推旧胎的等效胎龄,最后统一拟合。"""
    binned = df_comp.groupby("TyreAge")["PerfPct"].agg(
        ["median", "mean", "std", "count"]).reset_index()
    binned = binned[binned["count"] >= 3]
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
    binned = binned[binned["count"] >= 3]
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

    # 衰减率:胎龄>5 的线性回归斜率
    deg_mask = t_data > 5
    if deg_mask.sum() >= 5:
        slope, _ = np.polyfit(t_data[deg_mask], p_data[deg_mask], 1)
        deg_rate = -slope  # 正数 = 每圈衰减 %
    else:
        deg_rate = 0.0

    print(f"    [{compound}] 样本{int(binned['count'].sum())}  "
          f"P0={P0:.1f}%  峰值={peak_val:.1f}%@{t_peak:.0f}圈  "
          f"衰减={deg_rate:.3f}%/圈  范围={p_data.min():.1f}~{p_data.max():.1f}%")

    return {
        "P0": P0, "t_peak": t_peak, "peak_val": peak_val,
        "deg_rate": deg_rate,
        "t": t_data, "p": p_data, "trend": trend, "std": p_std,
        "n": int(binned["count"].sum()),
    }


def perf_pct_from_params(t, params):
    """从拟合参数计算性能保持率(%)——供 DP 使用。"""
    P0, peak_val = params["P0"], params["peak_val"]
    t_peak, deg_rate = params["t_peak"], params["deg_rate"]
    if t <= t_peak:
        return P0 + (peak_val - P0) * (t / max(t_peak, 1))
    return peak_val - deg_rate * (t - t_peak)


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


def fit_per_driver(df, pooled_results):
    """按车手拟合衰减曲线(样本<15 时退回全场曲线)。"""
    driver_results = {}
    for driver in sorted(df["Driver"].unique()):
        driver_results[driver] = {}
        for comp in VALID_COMPOUNDS:
            sub = df[(df["Driver"] == driver) & (df["AbsCompound"] == comp)]
            try:
                result = fit_one_compound(sub, comp)
                result["source"] = "driver"
            except (IndexError, ValueError):
                result = dict(pooled_results[comp])
                result["source"] = "pooled"
                result["n"] = len(sub)
            driver_results[driver][comp] = result
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
    df = build_clean_table(raw)
    print(f"  最终分析表: {len(df)} 行")
    print(f"  各配方样本: {df['AbsCompound'].value_counts().to_dict()}")

    print("\n[3] 拟合三阶段模型(思路B+暖胎恢复:出场圈修正后参与拟合) ...")
    fit_results = {}
    for comp in VALID_COMPOUNDS:
        sub = df[df["AbsCompound"] == comp]
        if len(sub) < 10:
            print(f"    [{comp}] 样本不足({len(sub)}),跳过")
            continue
        fit_results[comp] = fit_one_compound(sub, comp)

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

