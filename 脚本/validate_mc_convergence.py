#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蒙特卡洛收敛性验证
==================
固定赛中场景(Undercut·对手跟进),对模拟次数 N 做重复实验,
展示成功率估计的收敛性与方差衰减:

    N ∈ {100, 300, 1k, 3k, 10k, 30k, 100k},每个 N 重复 R 次(不同种子)

输出:
    图表/16_蒙特卡洛收敛图.csv   各 N 的均值/标准差/分位数/理论 SE
    图表/16_蒙特卡洛收敛图.png   收敛带状图(2.5~97.5% 分位带)+ 理论 1.96SE 曲线
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

# 固定场景: Undercut · 对手跟进(双方换新 C4,33 圈,落后 1.5s)
N_GRID = [100, 300, 1000, 3000, 10000, 30000, 100000]
REPS = [40, 40, 30, 20, 15, 10, 6]
FORM_STD = 0.20


def build_profiles():
    off = engine.DEFAULT_PARAMS["compound_offset"]
    bl = engine.STORE.base_time("LEC")
    bv = engine.STORE.base_time("VER")
    lm = engine.lap_profile("C4", 0, 33, bl, engine.STORE.deg_params("LEC", "C4"), off)
    lr = engine.lap_profile("C4", 0, 33, bv, engine.STORE.deg_params("VER", "C4"), off)
    return lm, lr, engine.driver_noise("LEC"), engine.driver_noise("VER")


def one_run(n, seed, lm, lr, s_me, s_rv):
    rng = np.random.default_rng(seed)
    t0m = engine.sample_pit_times(rng, n, 16.5, engine.DEFAULT_PARAMS)
    t0r = engine.sample_pit_times(rng, n, 16.5, engine.DEFAULT_PARAMS) + (-1.5)
    s, *_ = engine._run_branch(rng, n, 33, lm, lr, t0m, t0r,
                               s_me, s_rv, FORM_STD, FORM_STD)
    return float(s.mean()) * 100.0


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    print("=" * 55)
    print(" 蒙特卡洛收敛性验证 (Undercut·对手跟进)")
    print("=" * 55)
    lm, lr, s_me, s_rv = build_profiles()

    rows = []
    for n, r in zip(N_GRID, REPS):
        vals = np.array([one_run(n, 1000 * n + k, lm, lr, s_me, s_rv)
                         for k in range(r)])
        m = float(vals.mean())
        sd = float(vals.std(ddof=1)) if r > 1 else 0.0
        p_lo, p_hi = (float(np.percentile(vals, 2.5)),
                      float(np.percentile(vals, 97.5)))
        # 理论二项 SE(以均值成功率近似)
        p_frac = np.clip(m / 100.0, 1e-6, 1 - 1e-6)
        se_theory = 100.0 * float(np.sqrt(p_frac * (1 - p_frac) / n))
        rows.append({"N": n, "重复次数": r, "成功率均值(%)": round(m, 3),
                     "重复间标准差(pp)": round(sd, 3),
                     "P2.5(%)": round(p_lo, 2), "P97.5(%)": round(p_hi, 2),
                     "经验SE(pp)": round(sd, 3),
                     "理论SE(pp)": round(se_theory, 3),
                     "均值-参考(pp)": None})
        print(f"  N={n:>6} (R={r:>2}): mean={m:.2f}%  sd={sd:.3f}pp  "
              f"band=[{p_lo:.1f},{p_hi:.1f}]  理论SE={se_theory:.3f}pp")

    # 参考真值 = 最大 N 的均值
    ref = rows[-1]["成功率均值(%)"]
    for row in rows:
        row["均值-参考(pp)"] = round(row["成功率均值(%)"] - ref, 3)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(CHART_DIR, "16_蒙特卡洛收敛图.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"已保存: {csv_path}")

    n_arr = df["N"].to_numpy(float)
    mean = df["成功率均值(%)"].to_numpy(float)
    lo = df["P2.5(%)"].to_numpy(float)
    hi = df["P97.5(%)"].to_numpy(float)
    se = df["理论SE(pp)"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.fill_between(n_arr, lo, hi, color="#3f8cff", alpha=0.18,
                    label="重复实验 2.5~97.5% 分位带")
    ax.plot(n_arr, mean, "o-", color="#3f8cff", lw=2, label="成功率均值")
    ax.axhline(ref, color="#e10600", ls="--", lw=1.2,
               label=f"参考值(N={N_GRID[-1]}, {ref:.2f}%)")
    ax.plot(n_arr, ref + 1.96 * se, ":", color="#23d18b",
            label="参考值 ± 1.96×理论SE")
    ax.plot(n_arr, ref - 1.96 * se, ":", color="#23d18b")
    ax.set_xscale("log")
    ax.set_xlabel("模拟次数 N(对数轴)")
    ax.set_ylabel("成功率估计(%)")
    ax.set_title("蒙特卡洛收敛性: 成功率估计随 N 的收敛与方差衰减")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=9)
    fig.tight_layout()
    png_path = os.path.join(CHART_DIR, "16_蒙特卡洛收敛图.png")
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"已保存: {png_path}")


if __name__ == "__main__":
    main()
