#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交叉年份验证(消除 hindsight bias)
=================================
用早年数据拟合轮胎衰减模型,在留出年份上检验预测力:

    划分 A: 训练 2022+2023 → 验证 2024
    划分 B: 训练 2022+2023+2024 → 验证 2025

检验内容:
  1. 分年份拟合参数的稳定性(deg_rate / peak / t_peak 跨年对比);
  2. 训练参数对验证年"性能保持率-胎龄"分箱中位数的预测误差(MAE/RMSE/R²);
  3. 策略稳健性: 训练年参数解 DP 最优策略 vs 验证年自身参数求解,比较停站分割。

产出: 图表/15_交叉年份验证.csv 与 15_交叉年份验证.png
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
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "web"))

import fit_tire_degradation as ftd  # noqa: E402
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

COMPOUNDS = ["C5", "C4", "C3"]
COLORS = {"C5": "#e10600", "C4": "#ffd24d", "C3": "#b8bdc7"}
SPLITS = [
    ("A: 2022+2023 → 2024", [2022, 2023], 2024),
    ("B: 2022+2023+2024 → 2025", [2022, 2023, 2024], 2025),
]


def _prep(df_comp):
    """验证用数据规整: 胎龄取整、全部按新胎口径(跳过旧胎反推分支,
    避免小样本子集下分数胎龄导致分箱碎裂)。"""
    d = df_comp.copy()
    d["TyreAge"] = d["TyreAge"].round(0)
    if "FreshTyre" in d.columns:
        d["FreshTyre"] = True
    return d


def fit_params(df_comp):
    """拟合单配方(统计量法,与 fit_tire_degradation 的产出口径一致,
    但分箱计数阈值放宽以兼容小样本年份子集)。返回 4 参数字典或 None。"""
    d = _prep(df_comp)
    binned = d.groupby("TyreAge")["PerfPct"].agg(["median", "count"]).reset_index()
    binned = binned[binned["count"] >= 2]
    if len(binned) < 3:
        print(f"    [跳过] 样本不足({len(binned)} 个胎龄点)")
        return None
    t = binned["TyreAge"].to_numpy(float)
    p = binned["median"].to_numpy(float)
    window = min(5, len(p))
    trend = (pd.Series(p).rolling(window, center=True, min_periods=1)
             .median().to_numpy() if window >= 3 else p.copy())
    warm = t <= 2
    P0 = float(p[warm].min()) if warm.any() else float(p[0])
    peak_idx = int(np.argmax(trend))
    t_peak, peak_val = float(t[peak_idx]), float(trend[peak_idx])
    deg_mask = t > 5
    if deg_mask.sum() >= 5:
        slope, _ = np.polyfit(t[deg_mask], p[deg_mask], 1)
        deg = float(-slope)
    else:
        deg = 0.0
    return {"P0": P0, "t_peak": t_peak, "peak": peak_val, "peak_val": peak_val,
            "deg": deg, "deg_rate": deg, "n": int(binned["count"].sum())}


def eval_predict(params, df_valid_comp):
    """用 params 预测验证年分箱中位数,返回误差指标与分箱数据。"""
    d = _prep(df_valid_comp)
    binned = d.groupby("TyreAge")["PerfPct"].median().reset_index()
    binned = binned[binned["TyreAge"] <= 45]
    if len(binned) < 3:
        return None, None
    t = binned["TyreAge"].to_numpy(float)
    y = binned["PerfPct"].to_numpy(float)
    yhat = np.array([ftd.perf_pct_from_params(ti, params) for ti in t])
    resid = y - yhat
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")
    rng_y = float(y.max() - y.min())
    metrics = {"MAE_pct": round(float(np.abs(resid).mean()), 3),
               "MAE_s_per_lap": round(float(np.abs(resid).mean()) / 100 * 76.0, 3),
               "RMSE_pct": round(float(np.sqrt((resid ** 2).mean())), 3),
               "R2": round(r2, 4),
               "MAE_over_range": (round(float(np.abs(resid).mean()) / rng_y, 3)
                                  if rng_y > 1e-9 else None),
               "n_bins": len(binned)}
    return metrics, (t, y, yhat)


def dp_split_with(params_by_compound, driver="LEC"):
    """用给定参数解 1 停最优(全新 C5/C4/C3 各一套),返回 (总秒数, 策略文本)。"""
    base = engine.STORE.base_time(driver)
    offsets = engine.DEFAULT_PARAMS["compound_offset"]
    groups = [{"compound": c, "wear": 0, "count": 1,
               "sets": [{"id": f"N-{c}", "source": "新胎"}]}
              for c in COMPOUNDS]
    tables = []
    for g in groups:
        p = params_by_compound.get(g["compound"])
        tt = engine.lap_profile(g["compound"], 0, engine.TOTAL_LAPS,
                                base, p, offsets)
        tables.append(np.concatenate([[0.0], np.cumsum(tt)]))
    total, plan = engine._solve_inventory(groups, tables, 1, max_stops=1)
    if plan is None:
        return None, None
    stints = engine._plan_to_stints(plan, groups)
    txt = " → ".join(f"{s['compound']}({s['laps']}圈)" for s in stints)
    return round(total, 1), txt


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    print("=" * 55)
    print(" 交叉年份验证 (2022-2023→2024 / 2022-2024→2025)")
    print("=" * 55)

    raw = ftd.load_all_laps()
    df = ftd.build_clean_table(raw)

    rows = []
    curves = {}

    # ---- ① 分年份拟合参数稳定性
    print("\n[① 分年份拟合]")
    for year in (2022, 2023, 2024, 2025):
        for comp in COMPOUNDS:
            sub = df[(df["Year"] == year) & (df["AbsCompound"] == comp)]
            if len(sub) < 30:
                rows.append({"划分": f"分年 {year}", "配方": comp,
                             "项": "样本不足", "值": len(sub)})
                continue
            p = fit_params(sub)
            if p is None:
                rows.append({"划分": f"分年 {year}", "配方": comp,
                             "项": "拟合失败", "值": "—"})
                continue
            for k, label in (("P0", "P0"), ("t_peak", "t_peak"),
                             ("peak", "peak"), ("deg", "deg_rate"), ("n", "样本")):
                rows.append({"划分": f"分年 {year}", "配方": comp, "项": label,
                             "值": round(p[k], 4)})
            print(f"  {year} {comp}: deg={p['deg']:.4f}%/圈 "
                  f"peak={p['peak']:.2f}@{p['t_peak']:.0f} (n={p['n']})")

    # ---- ② 训练→验证 预测误差 + ③ 策略稳健性
    for split_name, train_years, valid_year in SPLITS:
        print(f"\n[②③ {split_name}]")
        p_tr_all, p_va_all = {}, {}
        for comp in COMPOUNDS:
            sub_tr = df[(df["Year"].isin(train_years)) & (df["AbsCompound"] == comp)]
            sub_va = df[(df["Year"] == valid_year) & (df["AbsCompound"] == comp)]
            if len(sub_tr) < 30 or len(sub_va) < 30:
                print(f"  {comp}: 样本不足,跳过")
                continue
            p_tr = fit_params(sub_tr)
            p_va = fit_params(sub_va)
            if p_tr is None or p_va is None:
                continue
            p_tr_all[comp], p_va_all[comp] = p_tr, p_va
            m, curve = eval_predict(p_tr, sub_va)
            m_in, _ = eval_predict(p_va, sub_va)
            if m:
                for k, v in (("验证MAE(%)", m["MAE_pct"]),
                             ("验证MAE(秒/圈)", m["MAE_s_per_lap"]),
                             ("验证RMSE(%)", m["RMSE_pct"]),
                             ("验证R2", m["R2"]),
                             ("MAE/数据极差", m["MAE_over_range"]),
                             ("验证R2(自身对照)", m_in["R2"] if m_in else None),
                             ("训练deg_rate", round(p_tr["deg"], 4)),
                             (f"{valid_year}deg_rate", round(p_va["deg"], 4))):
                    rows.append({"划分": split_name, "配方": comp, "项": k, "值": v})
                print(f"  {comp}: MAE={m['MAE_pct']:.3f}%({m['MAE_s_per_lap']:.2f}s/圈) "
                      f"RMSE={m['RMSE_pct']:.3f}% R2={m['R2']:.3f} "
                      f"(自身对照 {m_in['R2']:.3f})")
                curves[(split_name, comp)] = (curve, p_tr)
        if len(p_tr_all) == len(COMPOUNDS):
            t_tr, s_tr = dp_split_with(p_tr_all)
            t_va, s_va = dp_split_with(p_va_all)
            rows.append({"划分": split_name, "配方": "—", "项": "DP策略(训练参数)",
                         "值": s_tr})
            rows.append({"划分": split_name, "配方": "—", "项": "DP策略(验证年参数)",
                         "值": s_va})
            rows.append({"划分": split_name, "配方": "—", "项": "策略一致",
                         "值": "是" if s_tr == s_va else "否"})
            rows.append({"划分": split_name, "配方": "—",
                         "项": "总用时差(训练-验证,秒)",
                         "值": round((t_tr or 0) - (t_va or 0), 1)})
            print(f"  DP: 训练={s_tr} | 验证={s_va} | "
                  f"一致={'是' if s_tr == s_va else '否'}")

    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(CHART_DIR, "15_交叉年份验证.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {csv_path}")

    # ---- 图: 三配方 × 两划分
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for i, (split_name, _, valid_year) in enumerate(SPLITS):
        for j, comp in enumerate(COMPOUNDS):
            ax = axes[i][j]
            key = (split_name, comp)
            if key not in curves:
                ax.set_title(f"{comp} · 样本不足")
                continue
            (t, y, yhat), p_tr = curves[key]
            ax.scatter(t, y, s=26, color=COLORS[comp], alpha=0.85,
                       label=f"{valid_year}年实测(分箱中位)", zorder=3)
            tt = np.linspace(0, max(float(t.max()), 5), 100)
            yy = [ftd.perf_pct_from_params(ti, p_tr) for ti in tt]
            ax.plot(tt, yy, color="#3f8cff", lw=2, label="训练年拟合曲线(预测)")
            ax.set_title(f"{comp} · {split_name}", fontsize=10)
            ax.set_xlabel("胎龄(圈)")
            if j == 0:
                ax.set_ylabel("性能保持率(%)")
            ax.grid(alpha=0.3, ls=":")
            ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("交叉年份验证: 早年拟合曲线对留出年份的预测力(摩纳哥正赛)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = os.path.join(CHART_DIR, "15_交叉年份验证.png")
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"已保存: {png_path}")


if __name__ == "__main__":
    main()

