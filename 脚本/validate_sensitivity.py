#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
敏感性分析
==========
按任务清单口径做参数扰动实验,产出龙卷风图:

    进站损失 ±5s | 磨损系数 ±50% | 换胎时间均值 ±1s
    (附加) 配方偏移差 ±0.5s | 圈速波动 σ ×1.5

度量:
    DP 侧 = 最优 1 停总用时变化 Δ(秒) 与策略分割是否改变
    MC 侧 = Undercut·对手跟进 成功率变化 Δ(pp)

产出: 图表/17_敏感性分析.csv / 17_敏感性分析.png
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
sys.path.insert(0, os.path.join(PROJECT_DIR, "web"))
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

N_MC = 20000
FORM_STD = 0.20
COMPounds = ["C5", "C4", "C3"]

# (名称, DP侧修改, MC侧修改) —— 修改量以字典描述
CASES = [
    ("进站损失 −5s", {"pit": -5.0}, {"transit": -5.0}),
    ("进站损失 +5s", {"pit": +5.0}, {"transit": +5.0}),
    ("磨损系数 −50%", {"deg": 0.5}, {"deg": 0.5}),
    ("磨损系数 +50%", {"deg": 1.5}, {"deg": 1.5}),
    ("换胎均值 −1s", {"pit": -1.0}, {"change_mean": -1.0}),
    ("换胎均值 +1s", {"pit": +1.0}, {"change_mean": +1.0}),
    ("配方偏移差 −0.5s", {"off": -0.5}, {"off": -0.5}),
    ("配方偏移差 +0.5s", {"off": +0.5}, {"off": +0.5}),
    ("波动 σ ×1.5", {}, {"noise": 1.5}),
]


def scaled_params(code, comp, deg_mult=1.0):
    dp = dict(engine.STORE.deg_params(code, comp))
    dp["deg"] = dp["deg"] * deg_mult
    return dp


def dp_total(deg_mult=1.0, pit_delta=0.0, off_delta=0.0):
    """库存全新 C5/C4/C3 各一套的最优 1 停总用时与策略文本。"""
    base = engine.STORE.base_time("LEC")
    off = dict(engine.DEFAULT_PARAMS["compound_offset"])
    for c in off:
        if c == "C5":
            off[c] = off[c] + off_delta
        elif c == "C3":
            off[c] = off[c] - off_delta
    groups = [{"compound": c, "wear": 0, "count": 1,
               "sets": [{"id": f"N-{c}", "source": "新胎"}]} for c in COMPounds]
    tables = []
    for g in groups:
        p = scaled_params("LEC", g["compound"], deg_mult)
        tt = engine.lap_profile(g["compound"], 0, engine.TOTAL_LAPS, base, p, off)
        tables.append(np.concatenate([[0.0], np.cumsum(tt)]))
    saved = engine.PRERACE_PIT_LOSS
    engine.PRERACE_PIT_LOSS = saved + pit_delta
    try:
        total, plan = engine._solve_inventory(groups, tables, 1, max_stops=1)
    finally:
        engine.PRERACE_PIT_LOSS = saved
    if plan is None:
        return None, None
    stints = engine._plan_to_stints(plan, groups)
    txt = " → ".join(f"{s['compound']}({s['laps']}圈)" for s in stints)
    return round(total, 1), txt


def mc_success(deg_mult=1.0, transit_delta=0.0, change_mean_delta=0.0,
               off_delta=0.0, noise_mult=1.0, seed=20261001):
    """Undercut·对手跟进(双方换新 C4,33 圈,落后 1.5s)的成功率。"""
    off = dict(engine.DEFAULT_PARAMS["compound_offset"])
    for c in off:
        if c == "C5":
            off[c] = off[c] + off_delta
        elif c == "C3":
            off[c] = off[c] - off_delta
    bl = engine.STORE.base_time("LEC")
    bv = engine.STORE.base_time("VER")
    lm = engine.lap_profile("C4", 0, 33, bl, scaled_params("LEC", "C4", deg_mult), off)
    lr = engine.lap_profile("C4", 0, 33, bv, scaled_params("VER", "C4", deg_mult), off)
    p = dict(engine.DEFAULT_PARAMS)
    p["pit_normal_mean"] = p["pit_normal_mean"] + change_mean_delta
    transit = 16.5 + transit_delta
    s_me = engine.driver_noise("LEC") * noise_mult
    s_rv = engine.driver_noise("VER") * noise_mult
    fs = FORM_STD * noise_mult
    rng = np.random.default_rng(seed)
    t0m = engine.sample_pit_times(rng, N_MC, transit, p)
    t0r = engine.sample_pit_times(rng, N_MC, transit, p) - 1.5
    s, *_ = engine._run_branch(rng, N_MC, 33, lm, lr, t0m, t0r,
                               s_me, s_rv, fs, fs)
    return round(float(s.mean()) * 100.0, 2)


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    print("=" * 55)
    print(" 敏感性分析 (进站损失/磨损系数/换胎均值/偏移/波动)")
    print("=" * 55)

    dp0, s0 = dp_total()
    mc0 = mc_success()
    print(f"基准: DP 1停总用时 {dp0}s 策略[{s0}] | MC 成功率 {mc0}%")

    rows = [{"项": "基准", "DP总用时(秒)": dp0, "DP策略": s0,
             "DP变化(秒)": 0.0, "MC成功率(%)": mc0, "MC变化(pp)": 0.0}]
    for name, dp_mod, mc_mod in CASES:
        dp_v, s_v = dp_total(deg_mult=dp_mod.get("deg", 1.0),
                             pit_delta=dp_mod.get("pit", 0.0),
                             off_delta=dp_mod.get("off", 0.0))
        mc_v = mc_success(deg_mult=mc_mod.get("deg", 1.0),
                          transit_delta=mc_mod.get("transit", 0.0),
                          change_mean_delta=mc_mod.get("change_mean", 0.0),
                          off_delta=mc_mod.get("off", 0.0),
                          noise_mult=mc_mod.get("noise", 1.0))
        rows.append({"项": name,
                     "DP总用时(秒)": dp_v, "DP策略": s_v,
                     "DP变化(秒)": round((dp_v or 0) - (dp0 or 0), 1),
                     "MC成功率(%)": mc_v,
                     "MC变化(pp)": round(mc_v - mc0, 2)})
        print(f"  {name:<14} DP {dp_v}s({(dp_v or 0)-(dp0 or 0):+.1f}s, "
              f"{'策略变' if s_v != s0 else '策略不变'}) | MC {mc_v}%({mc_v-mc0:+.2f}pp)")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(CHART_DIR, "17_敏感性分析.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"已保存: {csv_path}")

    # ---- 龙卷风图
    cases = df[df["项"] != "基准"].copy()
    cases["dp_d"] = cases["DP变化(秒)"].astype(float)
    cases["mc_d"] = cases["MC变化(pp)"].astype(float)
    cases = cases.sort_values("mc_d", key=lambda c: c.abs())
    labels = cases["项"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, col, base_txt, title in (
            (axes[0], "dp_d", f"基准 {dp0}s", "DP 最优 1 停总用时变化 Δ(秒)"),
            (axes[1], "mc_d", f"基准 {mc0}%", "MC 成功率变化 Δ(百分点)")):
        vals = cases[col].to_numpy(float)
        colors = ["#3f8cff" if v >= 0 else "#e10600" for v in vals]
        ax.barh(labels, vals, color=colors, alpha=0.85)
        ax.axvline(0, color="#888", lw=1)
        for y, v in enumerate(vals):
            ax.text(v + (0.02 * max(abs(vals.max()), 1e-6)), y,
                    f"{v:+.2f}", va="center", fontsize=8.5)
        ax.set_title(title + f"  ({base_txt})", fontsize=10.5)
        ax.grid(alpha=0.25, ls=":", axis="x")
    fig.suptitle("敏感性分析(龙卷风图): 参数扰动对策略总用时与成功率的影响",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = os.path.join(CHART_DIR, "17_敏感性分析.png")
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"已保存: {png_path}")


if __name__ == "__main__":
    main()

