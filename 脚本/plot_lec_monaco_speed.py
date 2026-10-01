#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
勒克莱尔 2024 年摩纳哥站(正赛)每一圈的时速曲线绘制脚本。

数据来源
    F1毕设\2024\LEC\lec_monaco_2024_all_laps_telemetry.csv
    (由 fastf1 的 lap.get_telemetry() 汇总导出,含 Speed(km/h)、Distance(m)、LapNumber 等列)

用法
    python plot_lec_monaco_speed.py                     # 生成全部汇总图
    python plot_lec_monaco_speed.py --per-lap           # 额外为每一圈单独导出 PNG
    python plot_lec_monaco_speed.py --x time            # 横轴改成"圈内用时(s)"
    python plot_lec_monaco_speed.py --csv xxx.csv --outdir out --dpi 120

路径说明(2026-09 整理到 F1毕设 后更新)
    数据按"年 → 车手"两级存放于 F1毕设\ 下:
        F1毕设\<年份>\<车手>\<车手小写>_monaco_<年份>_all_laps_telemetry.csv
        例如 F1毕设\2024\LEC\lec_monaco_2024_all_laps_telemetry.csv
    图表直接输出到 CSV 所在目录,与运行时所在目录无关。
    可用 --csv / --outdir 显式指定;多个遥测 CSV 时会打印实际使用的文件。

生成的图(默认输出到 CSV 所在目录,即 F1毕设\<年份>\<车手>\)
    01_全部圈速度曲线叠加.png     所有圈的速度-圈内距离曲线叠加,按圈数着色
    02_每圈速度曲线网格.png       每圈一个小图,一图看完全部圈
    03_速度热力图.png             圈数(纵轴) x 圈内距离(横轴) 的速度热力图
    04_最快圈与最慢圈对比.png     最快 3 圈 / 最慢 3 圈 + 全程平均曲线
    05_每圈平均与最高速度.png     各圈平均速度(柱)与最高速度(线)随圈数变化
    laps/lap_XX.png               (--per-lap)每圈单独的时速曲线图
    lap_summary.csv               各圈统计(时长、最高/平均速度、静止时长、是否异常)

数据说明与清洗
    * 本 CSV 中没有第 2 圈(第 1 圈发车后发生事故,红旗中断后重新发车),因此圈号会跳过 2,
      实际有效圈数 77 圈。
    * 第 1 圈是红旗圈:遥测片段长达 2456 s,其中约 2250 s 赛车静止不动(停在 3201 m
      附近等待重新发车)。脚本会自动剔除"持续超过 15 s 的静止段",把该圈还原成接近
      正常一圈的形态,并在图例中单独标注(该圈时长仍偏长,属正常现象)。
    * CSV 里每圈的 Distance 已按圈重置,脚本再做一次归零,横轴即"圈内行驶距离"。
    * Speed 单位是 km/h(FastF1 原始单位),最大值 288 km/h,与摩纳哥赛道特征吻合。
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # 只存图,不开窗口,便于无界面批量运行
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------------------------------------------------------- 基本配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # ...\F1毕设\脚本
PROJECT_DIR = os.path.dirname(BASE_DIR)                    # ...\F1毕设
CSV_NAME = "lec_monaco_2024_all_laps_telemetry.csv"
DEFAULT_CSV = os.path.join(PROJECT_DIR, "2024", "LEC", CSV_NAME)

TIME_COL = "Time"               # 形如 "0 days 00:01:18.733000"
SPEED_COL = "Speed"             # km/h
DIST_COL = "Distance"           # 圈内距离 (m)
LAP_COL = "LapNumber"

STATIONARY_SPEED = 2.0          # 时速低于该值视为"几乎静止"(km/h)
STATIONARY_MIN_SECONDS = 15.0   # 静止超过该时长则判定为异常暂停(红旗/停车),自动剔除
TRACK_LENGTH = 3337.0           # 摩纳哥赛道单圈长度(m),仅用于横轴参考
Y_MAX = 320.0                   # 速度纵轴上限(km/h)

# ----------------------------------------------------------------- 车手中英文名映射
_DRIVER_NAMES = {
    "ALB": "阿尔本", "ALO": "阿隆索", "ANT": "安东内利",
    "BEA": "比尔曼", "BOR": "博尔托莱托", "BOT": "博塔斯", "COL": "科拉皮托",
    "DEV": "德弗里斯", "DOO": "杜汉", "GAS": "加斯利",
    "HAD": "哈贾尔", "HAM": "汉密尔顿", "HUL": "胡肯伯格",
    "LAT": "拉提菲", "LAW": "劳森", "LEC": "勒克莱尔",
    "MAG": "马格努森", "MSC": "米克·舒马赫", "NOR": "诺里斯",
    "OCO": "奥康", "PER": "佩雷兹", "PIA": "皮亚斯特里",
    "RIC": "里卡多", "RUS": "拉塞尔", "SAI": "塞恩斯",
    "SAR": "萨金特", "STR": "斯特罗尔", "TSU": "角田裕毅",
    "VER": "维斯塔潘", "VET": "维特尔", "ZHO": "周冠宇",
}
_DRIVER_CODE = "LEC"     # 由 main() 根据 CSV 路径自动设置
_DRIVER_NAME = "勒克莱尔"
_YEAR = 2024


def _init_driver_info(csv_path):
    """从 CSV 文件名中提取车手代号和年份,更新模块级变量。"""
    global _DRIVER_CODE, _DRIVER_NAME, _YEAR
    import re
    basename = os.path.basename(csv_path).lower()
    m = re.match(r"([a-z]+)_monaco_(\d{4})_", basename)
    if m:
        _DRIVER_CODE = m.group(1).upper()
        _YEAR = int(m.group(2))
    _DRIVER_NAME = _DRIVER_NAMES.get(_DRIVER_CODE, _DRIVER_CODE)


def resolve_csv(csv_path):
    r"""定位遥测 CSV。

    优先级:--csv 指定 → F1毕设\2024\LEC\ 默认文件 → F1毕设\<年份>\<车手>\ 下的同名文件
    → F1毕设\<年份>\<车手>\ 下第一个 *monaco*telemetry*.csv(按文件名排序取第一个)。
    """
    if csv_path:
        if os.path.exists(csv_path):
            return os.path.abspath(csv_path)
        sys.exit(f"指定的 CSV 不存在: {csv_path}")

    patterns = [DEFAULT_CSV,
                os.path.join(PROJECT_DIR, "*", "*", CSV_NAME),
                os.path.join(PROJECT_DIR, "*", "*", "*monaco*telemetry*.csv")]
    for pattern in patterns:
        if os.path.isfile(pattern):
            return os.path.abspath(pattern)
        matches = sorted(p for p in glob.glob(pattern) if os.path.isfile(p))
        if matches:
            if len(matches) > 1:
                print(f"[提示] 匹配到 {len(matches)} 个遥测 CSV,"
                      f"本次使用 {os.path.relpath(matches[0], PROJECT_DIR)}"
                      "(可用 --csv 指定其它文件)")
            return os.path.abspath(matches[0])
    sys.exit(r"找不到遥测 CSV。请把数据放到 F1毕设\<年份>\<车手>\ 下,"
             r"或用 --csv 指定文件路径。")


def resolve_outdir(out_dir, csv_path):
    r"""定位输出目录。未用 --outdir 指定时,图表输出到 CSV 所在目录。"""
    if out_dir:
        return out_dir
    return os.path.dirname(os.path.abspath(csv_path))


def set_chinese_font():
    """让 matplotlib 能正常显示中文与负号。"""
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        try:
            fm.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        break
    plt.rcParams["axes.unicode_minus"] = False


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="绘制各车手摩纳哥站每一圈的时速曲线(自动识别车手与年份)")
    p.add_argument("--csv", default=DEFAULT_CSV, help="遥测 CSV 路径")
    p.add_argument("--outdir", default="",
                   help=r"图片输出目录;留空则输出到 CSV 所在目录(F1毕设\<年份>\<车手>\)")
    p.add_argument("--x", choices=("distance", "time"), default="distance",
                   help="横轴: distance=圈内距离(m,默认), time=圈内用时(s)")
    p.add_argument("--per-lap", action="store_true",
                   help="额外为每一圈单独导出一张 PNG 到 <outdir>/laps/")
    p.add_argument("--dpi", type=int, default=150, help="图片分辨率,默认 150")
    p.add_argument("--stationary", type=float, default=STATIONARY_MIN_SECONDS,
                   help="静止超过多少秒判定为异常暂停并剔除,默认 15")
    return p.parse_args(argv)


# ----------------------------------------------------------------- 数据读取/清洗
def load_telemetry(csv_path):
    """读取 CSV,并把 Time 列解析为 timedelta。"""
    if not os.path.exists(csv_path):
        sys.exit(f"找不到数据文件: {csv_path}")
    usecols = [LAP_COL, TIME_COL, SPEED_COL, DIST_COL,
               "Throttle", "Brake", "nGear", "DRS"]
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df.dropna(subset=[LAP_COL, TIME_COL, SPEED_COL, DIST_COL]).copy()
    df[TIME_COL] = pd.to_timedelta(df[TIME_COL])
    df[LAP_COL] = df[LAP_COL].astype(int)
    df = df.sort_values([LAP_COL, TIME_COL], kind="stable").reset_index(drop=True)
    print(f"已读取 {len(df)} 行遥测数据,共 {df[LAP_COL].nunique()} 圈 "
          f"(圈号 {df[LAP_COL].min()} ~ {df[LAP_COL].max()})")
    return df


def _long_stationary_mask(speed, dt_seconds, min_seconds):
    """标记“时速<2 km/h 且连续时长超过 min_seconds”的行(如红旗停车)。"""
    stationary = speed < STATIONARY_SPEED
    mask = np.zeros(len(speed), dtype=bool)
    i, n = 0, len(speed)
    while i < n:
        if not stationary[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and stationary[j + 1]:
            j += 1
        if dt_seconds[i:j + 1].sum() >= min_seconds:
            mask[i:j + 1] = True
        i = j + 1
    return mask


def clean_lap(lap_df, min_stationary=STATIONARY_MIN_SECONDS):
    """清洗单圈遥测,返回绘图用的 x/t/speed 及统计信息。"""
    d = lap_df.reset_index(drop=True)
    t_raw = d[TIME_COL].dt.total_seconds().to_numpy(dtype=float)
    dt = np.diff(t_raw, prepend=t_raw[0])
    dt[0] = np.median(dt[1:]) if len(dt) > 1 else 0.0

    stop = _long_stationary_mask(d[SPEED_COL].to_numpy(dtype=float),
                                 dt, min_stationary)
    stopped_seconds = float(dt[stop].sum())
    interrupted = stopped_seconds > 0

    kept = d.loc[~stop]
    if len(kept) < 2:                     # 兜底:整圈几乎静止(理论上不会发生)
        kept = d.copy()

    # 用“有效行驶时间”作为时间轴:剔除暂停后,把断点按采样间隔拼接
    t_keep = kept[TIME_COL].dt.total_seconds().to_numpy(dtype=float)
    kdt = np.diff(t_keep, prepend=t_keep[0])
    kdt[0] = 0.0
    normal = kdt[(kdt > 0) & (kdt < 2.0)]
    fill = float(np.median(normal)) if normal.size else 0.13
    kdt = np.where((kdt > 2.0) | (kdt <= 0), fill, kdt)
    t = np.cumsum(kdt)

    speed = kept[SPEED_COL].to_numpy(dtype=float)
    dist = kept[DIST_COL].to_numpy(dtype=float)
    dist = dist - np.nanmin(dist)         # 圈内距离归零

    return {
        "lap": int(d[LAP_COL].iloc[0]),
        "x_distance": dist,
        "t": t,
        "speed": speed,
        "duration": float(t[-1]),         # 有效行驶时长(s)
        "v_max": float(np.nanmax(speed)),
        "v_mean": float(np.nanmean(speed)),
        "distance_total": float(dist[-1]),
        "stopped_seconds": stopped_seconds,
        "rows": len(d),
        "interrupted": interrupted,
    }


def build_laps(df, min_stationary=STATIONARY_MIN_SECONDS):
    """逐圈清洗,按圈号升序返回。"""
    laps = [clean_lap(g, min_stationary) for _, g in df.groupby(LAP_COL, sort=True)]
    laps.sort(key=lambda lap: lap["lap"])
    return laps


# ----------------------------------------------------------------- 绘图工具
def x_of(lap, x_axis):
    return lap["x_distance"] if x_axis == "distance" else lap["t"]


def x_label(x_axis):
    return "圈内行驶距离 (m)" if x_axis == "distance" else "圈内有效行驶时间 (s)"


# ----------------------------------------------------------------- 图 1
def plot_all_laps_overlay(laps, out_path, x_axis, dpi):
    """01 全部圈的速度曲线叠加(按圈数着色)。"""
    lap_nums = [lap["lap"] for lap in laps]
    cmap = plt.get_cmap("turbo")
    norm = mcolors.Normalize(vmin=min(lap_nums), vmax=max(lap_nums))

    fig, ax = plt.subplots(figsize=(13, 7))
    for lap in laps:
        special = lap["interrupted"]
        ax.plot(x_of(lap, x_axis), lap["speed"],
                lw=1.4 if special else 0.9,
                ls="--" if special else "-",
                color="black" if special else cmap(norm(lap["lap"])),
                alpha=1.0 if special else 0.55,
                zorder=6 if special else 2,
                label=(f"第 {lap['lap']} 圈(红旗圈,已剔除停车段)"
                       if special else None))
    ax.set_xlim(left=0)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel(x_label(x_axis))
    ax.set_ylabel("时速 (km/h)")
    ax.grid(alpha=0.3, ls=":")
    fig.suptitle(f"{_YEAR} 摩纳哥站 · {_DRIVER_NAME} · 各圈时速曲线(共 {len(laps)} 圈)",
                 fontsize=14)
    ax.set_title("颜色深浅 = 圈数;黑色虚线 = 红旗圈(第 1 圈)", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label("圈数")
    if any(lap["interrupted"] for lap in laps):
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print("  已保存:", os.path.basename(out_path))


# ----------------------------------------------------------------- 图 2
def plot_per_lap_grid(laps, out_path, x_axis, dpi, ncols=8):
    """02 逐圈小图网格:一图看完全部圈。"""
    n = len(laps)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.35 * ncols, 1.85 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for idx, lap in enumerate(laps):
        ax = axes[idx // ncols, idx % ncols]
        color = "crimson" if lap["interrupted"] else "#1f6fb2"
        ax.plot(x_of(lap, x_axis), lap["speed"], lw=0.9, color=color)
        ax.set_ylim(0, Y_MAX)
        ax.grid(alpha=0.3, ls=":")
        ax.set_title(f"第 {lap['lap']} 圈 | {lap['duration']:.1f} s | "
                     f"Vmax {lap['v_max']:.0f}",
                     fontsize=7, color=color,
                     fontweight="bold" if lap["interrupted"] else "normal")
    for idx in range(n, nrows * ncols):          # 多余的子图隐藏
        axes[idx // ncols, idx % ncols].axis("off")
    for c in range(ncols):
        axes[-1, c].set_xlabel("距离 (m)" if x_axis == "distance" else "时间 (s)",
                               fontsize=7)
    for r in range(nrows):
        axes[r, 0].set_ylabel("km/h", fontsize=7)
        axes[r, 0].tick_params(labelsize=6)
    for c in range(ncols):
        axes[-1, c].tick_params(labelsize=6)
    fig.suptitle(f"{_DRIVER_NAME} {_YEAR} 摩纳哥站 · 逐圈时速曲线(共 {n} 圈)"
                 "  每格标题: 圈号 | 有效行驶时长 | 最高时速(km/h)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print("  已保存:", os.path.basename(out_path))


# ----------------------------------------------------------------- 分箱工具
def _speed_matrix(laps, bin_size):
    """把各圈速度按“圈内距离”分箱求平均,返回 (圈数 x 分箱) 矩阵与分箱中心。"""
    max_dist = max(lap["distance_total"] for lap in laps)
    edges = np.arange(0.0, max_dist + bin_size, bin_size)
    centers = edges[:-1] + bin_size / 2
    matrix = np.full((len(laps), len(centers)), np.nan)
    for i, lap in enumerate(laps):
        idx = np.digitize(lap["x_distance"], edges) - 1
        ok = (idx >= 0) & (idx < len(centers))
        if not ok.any():
            continue
        grp = pd.Series(lap["speed"][ok]).groupby(idx[ok]).mean()
        matrix[i, grp.index.to_numpy(dtype=int)] = grp.to_numpy()
    return matrix, centers


def _mean_curve(laps, x_axis, bin_size):
    """所有给定圈在横轴分箱上的平均速度曲线。"""
    xmax = max(float(np.nanmax(x_of(lap, x_axis))) for lap in laps)
    edges = np.arange(0.0, xmax + bin_size, bin_size)
    centers = edges[:-1] + bin_size / 2
    acc = np.zeros(len(centers))
    cnt = np.zeros(len(centers))
    for lap in laps:
        idx = np.digitize(x_of(lap, x_axis), edges) - 1
        ok = (idx >= 0) & (idx < len(centers))
        np.add.at(acc, idx[ok], lap["speed"][ok])
        np.add.at(cnt, idx[ok], 1)
    return centers, np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


# ----------------------------------------------------------------- 图 3
def plot_speed_heatmap(laps, out_path, dpi, bin_size=20.0):
    """03 圈数 x 圈内距离的速度热力图。"""
    matrix, centers = _speed_matrix(laps, bin_size)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("lightgray")            # 无数据的距离段用灰底表示

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap,
                   vmin=0, vmax=Y_MAX, interpolation="nearest",
                   extent=(centers[0], centers[-1], 0, len(laps)))
    ax.set_xlabel("圈内行驶距离 (m)")
    ax.set_ylabel("圈数")
    ax.set_yticks(np.arange(len(laps)) + 0.5)
    ax.set_yticklabels([str(lap["lap"]) for lap in laps], fontsize=5)
    ax.tick_params(axis="y", length=0)
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.text(centers[-1] * 0.995, 0.55, "第 1 圈(红旗圈)", ha="right", va="center",
            fontsize=8, color="black")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("时速 (km/h)")
    fig.suptitle(f"{_DRIVER_NAME} {_YEAR} 摩纳哥站 · 速度热力图"
                 f"(每 {bin_size:.0f} m 一个分箱,灰色=无数据)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print("  已保存:", os.path.basename(out_path))


# ----------------------------------------------------------------- 图 4
def plot_fastest_slowest(laps, out_path, x_axis, dpi, top_n=3, bin_size=10.0):
    """04 最快若干圈 / 最慢若干圈 + 全部有效圈平均曲线。"""
    normal = [lap for lap in laps if not lap["interrupted"]]
    if not normal:
        print("  跳过最快/最慢圈对比:没有正常圈数据")
        return
    ranked = sorted(normal, key=lambda lap: lap["duration"])
    fastest = ranked[:top_n]
    slowest = ranked[-top_n:][::-1]
    centers, mean_curve = _mean_curve(normal, x_axis, bin_size)

    fast_colors = ["#1a9850", "#41ab5d", "#78c679", "#addd8e"]
    slow_colors = ["#b30000", "#e34a33", "#fc8d59", "#fdbb84"]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for k, lap in enumerate(fastest):
        ax.plot(x_of(lap, x_axis), lap["speed"], lw=1.7,
                color=fast_colors[k % len(fast_colors)],
                label=f"最快: 第 {lap['lap']} 圈 · {lap['duration']:.2f} s")
    for k, lap in enumerate(slowest):
        ax.plot(x_of(lap, x_axis), lap["speed"], lw=1.4, ls="-",
                color=slow_colors[k % len(slow_colors)],
                label=f"最慢: 第 {lap['lap']} 圈 · {lap['duration']:.2f} s")
    ax.plot(centers, mean_curve, color="black", lw=1.3, ls="--",
            label=f"全部有效圈平均({len(normal)} 圈)")

    ax.set_xlim(left=0)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel(x_label(x_axis))
    ax.set_ylabel("时速 (km/h)")
    ax.grid(alpha=0.3, ls=":")
    fig.suptitle(f"{_DRIVER_NAME} {_YEAR} 摩纳哥站 · 最快圈 vs 最慢圈时速曲线对比"
                 "(不含红旗圈)", fontsize=13)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print("  已保存:", os.path.basename(out_path))


# ----------------------------------------------------------------- 图 5
def plot_lap_speed_stats(laps, out_path, dpi):
    """05 各圈平均速度(柱)、最高速度(线)与有效行驶时长(右轴)。"""
    nums = [lap["lap"] for lap in laps]
    v_mean = [lap["v_mean"] for lap in laps]
    v_max = [lap["v_max"] for lap in laps]
    dur = [lap["duration"] for lap in laps]
    colors = ["crimson" if lap["interrupted"] else "#4c72b0" for lap in laps]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(nums, v_mean, color=colors, alpha=0.85, width=0.7,
           label="平均速度 (km/h)")
    ax.plot(nums, v_max, color="#2b3d63", lw=1.2, label="最高速度 (km/h)")
    ax.set_xlabel("圈数")
    ax.set_ylabel("速度 (km/h)")
    ax.set_ylim(0, Y_MAX)
    ax.set_xticks(np.arange(0, max(nums) + 1, 5))
    ax.grid(alpha=0.3, ls=":", axis="y")

    ax2 = ax.twinx()
    ax2.plot(nums, dur, color="darkorange", lw=1.5, marker="o", ms=3,
             label="有效行驶时长 (s)")
    ax2.set_ylabel("有效行驶时长 (s)", color="darkorange")
    ax2.tick_params(axis="y", colors="darkorange")
    ax2.set_ylim(60, max(dur) + 15)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{_DRIVER_NAME} {_YEAR} 摩纳哥站 · 各圈平均/最高时速与行驶时长"
                 "(红色柱 = 异常圈)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print("  已保存:", os.path.basename(out_path))


# ----------------------------------------------------------------- 逐圈单独导出
def export_individual_laps(laps, out_dir, x_axis, dpi):
    """--per-lap:每一圈单独导出一张时速曲线 PNG。"""
    lap_dir = os.path.join(out_dir, "laps")
    os.makedirs(lap_dir, exist_ok=True)
    for lap in laps:
        color = "crimson" if lap["interrupted"] else "#1f6fb2"
        x = x_of(lap, x_axis)
        speed = lap["speed"]

        fig, ax = plt.subplots(figsize=(11, 4.4))
        ax.plot(x, speed, color=color, lw=1.5)
        ax.fill_between(x, speed, color=color, alpha=0.15)
        ax.set_xlim(left=0)
        ax.set_ylim(0, Y_MAX)
        ax.set_xlabel(x_label(x_axis))
        ax.set_ylabel("时速 (km/h)")
        ax.grid(alpha=0.3, ls=":")
        ax.axhline(lap["v_mean"], color="gray", lw=0.9, ls=":",
                   label=f"平均 {lap['v_mean']:.0f} km/h")

        k = int(np.nanargmax(speed))
        ax.annotate(f"Vmax {speed[k]:.0f} km/h", xy=(x[k], speed[k]),
                    xytext=(8, -14), textcoords="offset points",
                    fontsize=8, color=color)
        ax.legend(loc="lower right", fontsize=8)

        parts = [f"有效行驶时长 {lap['duration']:.1f} s",
                 f"最高 {lap['v_max']:.0f} km/h",
                 f"平均 {lap['v_mean']:.0f} km/h",
                 f"行驶距离 {lap['distance_total']:.0f} m"]
        if lap["interrupted"]:
            parts.append(f"已剔除停车 {lap['stopped_seconds']:.0f} s(红旗)")
        fig.suptitle(f"{_DRIVER_NAME} {_YEAR} 摩纳哥站 · 第 {lap['lap']} 圈时速曲线",
                     fontsize=13, color=color)
        ax.set_title(" | ".join(parts), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(lap_dir, f"lap_{lap['lap']:02d}.png"), dpi=dpi)
        plt.close(fig)
    print(f"  已保存 {len(laps)} 张单圈图到: {lap_dir}")


# ----------------------------------------------------------------- 统计输出
def save_summary(laps, out_path):
    """把每圈的时长/最高速度/平均速度等统计写入 CSV。"""
    rows = [{
        "圈号": lap["lap"],
        "有效行驶时长_s": round(lap["duration"], 2),
        "最高时速_kmh": round(lap["v_max"], 1),
        "平均时速_kmh": round(lap["v_mean"], 1),
        "圈内行驶距离_m": round(lap["distance_total"], 1),
        "剔除停车_s": round(lap["stopped_seconds"], 1),
        "是否异常圈": "是" if lap["interrupted"] else "否",
        "遥测行数": lap["rows"],
    } for lap in laps]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("  已保存:", os.path.basename(out_path))
    return summary


# ----------------------------------------------------------------- 主程序
def main(argv=None):
    args = parse_args(argv)
    set_chinese_font()
    args.csv = resolve_csv(args.csv)
    _init_driver_info(args.csv)
    args.outdir = resolve_outdir(args.outdir, args.csv)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"数据文件: {args.csv}")
    print(f"输出目录: {os.path.abspath(args.outdir)}")

    df = load_telemetry(args.csv)
    laps = build_laps(df, args.stationary)
    print(f"清洗完成:共 {len(laps)} 圈;横轴 = "
          f"{'圈内行驶距离 (m)' if args.x == 'distance' else '圈内有效行驶时间 (s)'}")
    for lap in laps:
        if lap["interrupted"]:
            print(f"    [异常圈] 第 {lap['lap']} 圈:剔除停车 "
                  f"{lap['stopped_seconds']:.1f} s,有效行驶时长 "
                  f"{lap['duration']:.1f} s")
    lap_set = {lap["lap"] for lap in laps}
    missing = sorted(set(range(1, max(lap_set) + 1)) - lap_set)
    if missing:
        print(f"    [提示] 数据源中缺少圈号 {missing} 的遥测(红旗后重新发车)")

    print("开始绘图:")
    plot_all_laps_overlay(laps, os.path.join(args.outdir, "01_全部圈速度曲线叠加.png"),
                          args.x, args.dpi)
    plot_per_lap_grid(laps, os.path.join(args.outdir, "02_每圈速度曲线网格.png"),
                      args.x, min(args.dpi, 120))
    plot_speed_heatmap(laps, os.path.join(args.outdir, "03_速度热力图.png"), args.dpi)
    plot_fastest_slowest(laps, os.path.join(args.outdir, "04_最快圈与最慢圈对比.png"),
                         args.x, args.dpi)
    plot_lap_speed_stats(laps, os.path.join(args.outdir, "05_每圈平均与最高速度.png"),
                         args.dpi)
    summary = save_summary(laps, os.path.join(args.outdir, "lap_summary.csv"))
    if args.per_lap:
        export_individual_laps(laps, args.outdir, args.x, args.dpi)

    normal = summary[summary["是否异常圈"] == "否"].sort_values("有效行驶时长_s")
    print("\n统计摘要")
    print("    最快 5 圈:")
    for _, row in normal.head(5).iterrows():
        print(f"        第 {int(row['圈号']):>2} 圈  {row['有效行驶时长_s']:.2f} s"
              f"  最高时速 {row['最高时速_kmh']:.0f} km/h")
    print(f"    最快圈: 第 {int(normal.iloc[0]['圈号'])} 圈 · "
          f"{normal.iloc[0]['有效行驶时长_s']:.2f} s")
    print(f"    最慢圈(不含异常圈): 第 {int(normal.iloc[-1]['圈号'])} 圈 · "
          f"{normal.iloc[-1]['有效行驶时长_s']:.2f} s")
    print(f"    有效圈平均时长: {normal['有效行驶时长_s'].mean():.2f} s")
    print(f"    全场最高时速: {summary['最高时速_kmh'].max():.0f} km/h")
    print(f"\n全部图表已输出到: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
