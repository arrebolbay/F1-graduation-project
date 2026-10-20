#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
一键复现(run_all) —— 按依赖顺序跑完整条离线分析链路
====================================================

为什么要有这个文件
------------------
`图表/` 里的 20 多张图表是**一条有依赖的流水线**产出的:参数表(07/07b/07c)
决定 DP(08~10)、DP 决定 Stackelberg(11~13/19/20)、σ(14)决定蒙特卡洛的胜率。
以前靠人肉按顺序敲 8 个命令,漏跑一个就会出现"图表里新旧口径混用"——
这正是之前"同一份数据两个最优策略"能潜伏很久的原因之一。

用法
----
    python 脚本\run_all.py                 # 跑主链路(不含需要联网的抓数)
    python 脚本\run_all.py --with-export    # 额外重新抓取并导出原始数据(慢,需联网)
    python 脚本\run_all.py --with-charts    # 额外重画 2024 LEC 单圈图(2~3 分钟)
    python 脚本\run_all.py --only dp,stackelberg
    python 脚本\run_all.py --from stackelberg
    python 脚本\run_all.py --list           # 只看步骤表
    python 脚本\run_all.py --keep-going     # 出错也继续(默认中断)

约定
----
* 每个步骤是一次独立子进程,工作目录固定为项目根,日志同时写入
  ``日志/run_all_<时间戳>.log``;
* 任一步非零退出即中断并打印该步的日志尾部(除非 ``--keep-going``);
* 结束时输出"步骤 / 耗时 / 状态"一览,便于写论文附录复现说明。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(PROJECT_DIR, "日志")

# (键, 脚本, 额外参数, 说明, 是否默认执行)
STEPS = [
    ("clean-export", "batch_export.py", ["--years", "2022", "2023", "2024", "2025",
                                         "--skip-charts"],
     "抓取 + 导出遥测/圈状态 CSV(需联网, 首次最慢)", False),
    ("fit-tyre", "fit_tire_degradation.py", [],
     "轮胎衰减拟合 → 07 / 07b / 07c / 07d", True),
    ("fit-noise", "fit_lap_noise.py", [],
     "圈速波动 σ + 正态性检验 → 14 / 14b", True),
    ("dp", "solve_dp_strategy.py", [],
     "DP 赛前最优策略 → 08 / 09 / 10 系列", True),
    ("stackelberg", "solve_stackelberg.py", [],
     "Stackelberg 蒙特卡洛 + 收益矩阵 → 11 / 12 / 13 / 19 / 20", True),
    ("mc-convergence", "validate_mc_convergence.py", [],
     "蒙特卡洛收敛性 → 16", True),
    ("sensitivity", "validate_sensitivity.py", [],
     "敏感性分析 → 17", True),
    ("cross-year", "validate_cross_year.py", [],
     "交叉年份验证 → 15", True),
    ("history-2024", "validate_history_2024.py", [],
     "2024 红旗案例历史回溯 → 18", True),
    ("cleaning-check", "validate_cleaning.py", [],
     "清洗口径一致性验证 → 15b", True),
    ("charts-2024", "plot_lec_monaco_speed.py", [],
     "2024 LEC 时速曲线 + lap_summary(2~3 分钟)", False),
]


def _available():
    return [s for s in STEPS if os.path.exists(os.path.join(SCRIPT_DIR, s[1]))]


def select_steps(args):
    steps = _available()
    if args.only:
        want = [x.strip() for x in args.only.split(",") if x.strip()]
        unknown = [w for w in want if w not in {s[0] for s in steps}]
        if unknown:
            sys.exit(f"未知步骤: {', '.join(unknown)}(用 --list 查看)")
        steps = [s for s in steps if s[0] in want]
    else:
        if not args.with_export:
            steps = [s for s in steps if s[0] != "clean-export"]
        if not args.with_charts:
            steps = [s for s in steps if s[0] != "charts-2024"]
    if args.from_step:
        keys = [s[0] for s in steps]
        if args.from_step not in keys:
            sys.exit(f"--from 指定了不存在的步骤: {args.from_step}")
        steps = steps[keys.index(args.from_step):]
    return steps



def _tee(proc, log_file):
    """逐行转发子进程输出并写日志(不带缓冲,便于实时看进度)。"""
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()


def run_step(name, script, extra, desc, log_file):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script), *extra]
    header = f"\n{'=' * 68}\n[{name}] {desc}\n  $ {' '.join(cmd)}\n{'=' * 68}"
    print(header, flush=True)
    log_file.write(header + "\n")

    t0 = time.perf_counter()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    try:
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_DIR, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
    except OSError as exc:                     # pragma: no cover - 环境问题
        print(f"  启动失败: {exc}")
        return time.perf_counter() - t0, False, f"启动失败: {exc}"

    _tee(proc, log_file)
    code = proc.wait()
    dt = time.perf_counter() - t0
    ok = code == 0
    print(f"  → {'OK' if ok else 'FAILED'}  {dt:.1f}s", flush=True)
    log_file.write(f"  → {'OK' if ok else 'FAILED'}  {dt:.1f}s\n")
    return dt, ok, ("" if ok else f"退出码 {code}")



def main(argv=None):
    p = argparse.ArgumentParser(
        description="按依赖顺序复现全部离线图表(见 STEPS)")
    p.add_argument("--with-export", action="store_true",
                   help="包含 batch_export.py(联网抓数, 很慢)")
    p.add_argument("--with-charts", action="store_true",
                   help="包含 plot_lec_monaco_speed.py(2024 LEC 单圈图, 慢)")
    p.add_argument("--only", default="", help="只跑指定步骤, 逗号分隔")
    p.add_argument("--from", dest="from_step", default="",
                   help="从某步骤开始跑(含)")
    p.add_argument("--keep-going", action="store_true", help="某步失败也继续")
    p.add_argument("--list", action="store_true", help="只打印步骤表")
    args = p.parse_args(argv)

    if args.list:
        print(f"{'步骤':<16}{'默认':<6}{'脚本':<28}说明")
        for key, script, _extra, desc, default in _available():
            print(f"{key:<16}{'是' if default else '否':<6}{script:<28}{desc}")
        return 0

    steps = select_steps(args)
    if not steps:
        sys.exit("没有可执行的步骤(检查 --only / --from)")

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"run_all_{stamp}.log")

    print(f"项目根: {PROJECT_DIR}")
    print(f"Python : {sys.executable}")
    print(f"日志   : {log_path}")
    print(f"步骤   : {' → '.join(s[0] for s in steps)}")

    results, failed = [], []
    t_all = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"run_all @ {stamp}\nPython: {sys.executable}\n\n")
        for key, script, extra, desc, _default in steps:
            dt, ok, why = run_step(key, script, extra, desc, log_file)
            results.append((key, dt, ok, why))
            if not ok:
                failed.append(key)
                if not args.keep_going:
                    print(f"\n步骤 {key} 失败,已中断。"
                          f"加 --keep-going 可忽略错误继续。")
                    break

    total = time.perf_counter() - t_all
    print(f"\n{'=' * 68}\n汇总  (总耗时 {total:.1f}s)\n{'=' * 68}")
    for key, dt, ok, why in results:
        print(f"  {'OK ' if ok else 'BAD'} {key:<16}{dt:>7.1f}s  {why}")
    if failed:
        print(f"\n失败步骤: {', '.join(failed)}")
        print(f"完整日志: {log_path}")
        return 1
    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
