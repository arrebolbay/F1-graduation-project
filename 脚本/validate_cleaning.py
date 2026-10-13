#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
统一清洗口径验证(2024 摩纳哥红旗案例)
=====================================
验证 :mod:`laps_cleaning` 的判定在**全部车手之间一致**,
并确认「拟合链路剔除异常圈、模拟链路仍计入胎龄」两条链路都成立。

背景
----
2024 摩纳哥站第 1 圈发生严重事故并出示红旗,导致:
    第 1 圈  TrackStatus=1254(绿→黄→红旗→SC),全体进站,圈速约 2500 s
    第 2 圈  TrackStatus=14 / 41(SC 重启),圈速 112~148 s
    第 3 圈  TrackStatus=1(全绿),仍是车阵中的重启圈,圈速 84~90 s
    第 4 圈起 正常比赛圈,约 82~84 s

旧口径下 ``lap_summary.csv`` 的「是否异常圈」在车手间不一致
(ALB/ALO/GAS/RIC/STR/TSU/ZHO 标 [1,2],其余仅标 [1]),因为判据是
「遥测静止 > 15 s」,而第 2 圈遥测只对部分车手存在。

本脚本输出
    图表\15b_清洗口径验证_2024摩纳哥.csv  每车 L1~L4 的判定与原因
    控制台断言结果(全部车手必须一致)

用法
    python validate_cleaning.py
"""

import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PROJECT_DIR, "缓存", "f1_cache")
CHART_DIR = os.path.join(PROJECT_DIR, "图表")
OUT_CSV = os.path.join(CHART_DIR, "15b_清洗口径验证_2024摩纳哥.csv")

sys.path.insert(0, SCRIPT_DIR)
import laps_cleaning as lc  # noqa: E402

YEAR, EVENT, SESSION_TYPE = 2024, "Monaco", "R"
CHECK_LAPS = (1, 2, 3, 4, 5)


def load_laps():
    import fastf1
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    sess = fastf1.get_session(YEAR, EVENT, SESSION_TYPE)
    sess.load(telemetry=False, weather=False, messages=False)
    laps = sess.laps.copy()
    laps["Year"] = YEAR
    # 统一清洗口径需要的字段(与 fit_tire_degradation.build_clean_table 一致)
    laps["LapTimeS"] = laps["LapTime"].dt.total_seconds()
    return laps


def build_table(laps):
    """按统一口径打标记(含胎龄,证明"不删行" ⇒ 模拟链路胎龄不断链)。"""
    df = laps.dropna(subset=["LapNumber", "Driver"]).copy()
    df = df.sort_values(["Driver", "LapNumber"])
    if "TyreLife" in df.columns and df["TyreLife"].notna().any():
        df["TyreAge"] = df["TyreLife"].fillna(1) - 1
    else:
        df["TyreAge"] = (df.groupby(["Year", "Driver", "Stint"])
                           .cumcount())
    return lc.add_lap_flags(df, fuel_effect=lc.FUEL_EFFECT_SEC_PER_LAP)


def main():
    print("=" * 74)
    print(f" 统一清洗口径验证 —— {YEAR} {EVENT} {SESSION_TYPE}")
    print("=" * 74)

    raw = load_laps()
    df = build_table(raw)
    lc.summarise_flags(df, f"{YEAR} 摩纳哥 全站 ")

    # ── 每车 L1~L5 明细 ─────────────────────────────────────
    rows = []
    for drv in sorted(df["Driver"].unique()):
        dl = df[df["Driver"] == drv]
        for ln in CHECK_LAPS:
            hit = dl[dl["LapNumber"] == ln]
            if hit.empty:
                rows.append({"车手": drv, "圈": ln, "圈状态": "-",
                             "圈速_s": None, "胎龄": None,
                             "是否异常圈": lc.YES, "异常原因": "无该圈数据",
                             "参与拟合": False})
                continue
            r = hit.iloc[0]
            lt = r["LapTimeS"]
            rows.append({
                "车手": drv, "圈": int(ln), "圈状态": r["圈状态"],
                "圈速_s": round(float(lt), 1) if pd.notna(lt) else None,
                "胎龄": int(r["TyreAge"]) if pd.notna(r["TyreAge"]) else None,
                "是否异常圈": lc.NO if r["include_in_fit"] else lc.YES,
                "异常原因": r["exclude_reason"] or "-",
                "参与拟合": bool(r["include_in_fit"]),
            })
    detail = pd.DataFrame(rows)

    key = ["异常原因"]
    pivot = detail.pivot_table(index="圈", columns="车手", values=key,
                               aggfunc="first")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    print("\n[每车 L1~L5 异常原因(行=圈,列=车手)]")
    print(pivot.to_string())

    crosstab = (detail.assign(判定=detail["异常原因"])
                      .pivot_table(index="圈", columns="判定",
                                   values="车手", aggfunc="count")
                      .fillna(0).astype(int))
    print("\n[判定分布]")
    print(crosstab.to_string())

    # ── 一致性断言 ──────────────────────────────────────────
    # 确定性判据(发车/中性化/重启)必须全车手一致;
    # 离群判据是数据相对的统计判据,同一规则下允许逐车结果不同。
    EXPECTED = {1: lc.REASON_START, 2: lc.REASON_NEUTRAL, 3: lc.REASON_RESTART}
    print("\n[一致性检查] 确定性判据必须全车手一致")
    ok = True
    for ln in CHECK_LAPS:
        g = detail[detail["圈"] == ln]
        has = g[g["异常原因"] != "无该圈数据"]
        if has.empty:
            print(f"  L{ln}: 全部车手均无该圈数据(跳过)")
            continue
        reasons = sorted(has["异常原因"].unique())
        n_missing = int((g["异常原因"] == "无该圈数据").sum())
        expect = EXPECTED.get(ln)
        if expect is not None:
            good = (reasons == [expect])
            ok = ok and good
            print(f"  L{ln}: {'✅' if good else '★'} 期望「{expect}」"
                  f" —— 实际 {reasons}"
                  f"({len(has)} 车标注,{n_missing} 车无该圈数据)")
        else:
            print(f"  L{ln}: (统计判据)实际 {reasons}"
                  f"({len(has)} 车标注,{n_missing} 车无该圈数据)")

    # ── 两条链路 ────────────────────────────────────────────
    print("\n[两条链路]")
    fit = lc.fit_ready(df, drop_pit_out=True)
    sim = lc.sim_ready(df)
    print(f"  ① 拟合链路 fit_ready():   {len(fit)} 圈"
          f"(原始 {len(df)} 圈,剔除 {len(df) - len(fit)} 圈)")
    print(f"  ② 模拟链路 sim_ready():   {len(sim)} 圈 —— 不删行,"
          f"胎龄 TyreAge 连续累计")
    for drv in ("LEC", "VER", "TSU"):
        dl = df[df["Driver"] == drv].sort_values("LapNumber")
        if dl.empty:
            continue
        head = dl[dl["LapNumber"] <= 5]
        ages = ",".join(f"{int(a)}" for a in head["TyreAge"].tolist())
        reasons = "/".join(r if r else "拟合" for r in head["exclude_reason"])
        print(f"     {drv}: L1~L5 胎龄=[{ages}]  判定=[{reasons}]"
              "  → 红旗/SC 圈胎龄照样 +1")
    print("\n  注: DP(脚本\\solve_dp_strategy.py)与蒙特卡洛(web\\engine.py)"
          "的胎龄为逐圈递增计数器,\n      与圈速拟合是否剔除该圈无关 ⇒ "
          "模拟口径天然保留全部圈数。")

    os.makedirs(CHART_DIR, exist_ok=True)
    detail.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {OUT_CSV}")
    print("\n结论: " + ("✅ 全部车手判定一致" if ok else "❌ 存在不一致"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
