#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
清洗口径不变量测试(laps_cleaning)
=================================

2026-10 统一清洗口径的核心约定是"**打标记、不删行**":

* 拟合链路 :func:`laps_cleaning.fit_ready` 只取 ``include_in_fit`` 的子集;
* 模拟链路 :func:`laps_cleaning.sim_ready` 原样返回,**胎龄照常累计**。

如果哪天有人图省事在 :func:`add_lap_flags` 里直接 ``dropna()``,DP 的
``age + 1`` 就会和真实圈序错位,策略会以"少跑了几圈"的方式变好 —— 这种错误
不会报错,只会让结论悄悄变漂亮。所以这里用合成数据把语义钉死。
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("脚本", "web"):
    _p = os.path.join(ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import laps_cleaning as lc  # noqa: E402


def make_laps(n=24, sc_lap=10, pit_lap=12, outlier_lap=18):
    """构造一段干净的合成圈次: 只有 sc_lap / pit_lap / outlier_lap 三个异常。

    ``outlier_lap`` 必须是**绿旗圈** —— 离群判定只在 ``is_fit_candidate`` 上做,
    中性化圈已经被前面的判据排除了,不会被标成离群。
    """
    rng = np.random.default_rng(0)
    base = 76.0 + rng.normal(0, 0.05, n)
    ts = ["1"] * n
    ts[sc_lap - 1] = "4"                      # 安全车
    lap_time = base.copy()
    lap_time[0] += 3.0                        # 发车圈(静止起步)
    lap_time[pit_lap - 1] += 19.0             # 进站圈
    lap_time[outlier_lap - 1] += 9.0          # 绿旗圈上的明显离群(如小失误)
    pit_in = [pd.NaT] * n
    pit_in[pit_lap - 1] = pd.Timestamp("2024-05-26 15:10:00")
    return pd.DataFrame({
        "Year": 2024,
        "Driver": "LEC",
        "LapNumber": np.arange(1, n + 1),
        "LapTimeS": lap_time,
        "TrackStatus": ts,
        "Stint": [1] * pit_lap + [2] * (n - pit_lap),
        "TyreLife": list(range(1, pit_lap + 1)) + list(range(1, n - pit_lap + 1)),
        "Compound": ["MEDIUM"] * pit_lap + ["HARD"] * (n - pit_lap),
        "PitInTime": pit_in,
        "PitOutTime": [pd.NaT] * n,
    })


class TestFlags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = make_laps()
        cls.flagged = lc.add_lap_flags(cls.raw, fuel_effect=lc.FUEL_EFFECT_SEC_PER_LAP)

    def test_no_row_dropped(self):
        self.assertEqual(len(self.flagged), len(self.raw))
        self.assertEqual(self.flagged["LapNumber"].tolist(),
                         self.raw["LapNumber"].tolist())

    def test_green_flag(self):
        self.assertTrue(self.flagged.loc[1, "is_green"])          # 第 2 圈绿旗
        self.assertFalse(self.flagged.loc[9, "is_green"])         # 第 10 圈 SC
        self.assertTrue(self.flagged.loc[17, "is_green"])         # 第 18 圈绿旗(但离群)

    def test_start_lap_flagged(self):
        self.assertTrue(self.flagged.loc[0, "is_start_lap"])
        self.assertFalse(self.flagged.loc[0, "include_in_fit"])

    def test_pit_lap_flagged(self):
        self.assertTrue(self.flagged.loc[11, "is_pit_in"])
        self.assertFalse(self.flagged.loc[11, "include_in_fit"])

    def test_neutralised_excluded(self):
        self.assertFalse(self.flagged.loc[9, "include_in_fit"])
        self.assertTrue(self.flagged.loc[9, "exclude_reason"])

    def test_outlier_excluded(self):
        self.assertTrue(self.flagged.loc[17, "is_outlier"])
        self.assertFalse(self.flagged.loc[17, "include_in_fit"])

    def test_clean_lap_included(self):
        self.assertTrue(self.flagged.loc[4, "include_in_fit"])    # 第 5 圈
        self.assertEqual(self.flagged.loc[4, "exclude_reason"], "")

    def test_tyre_life_untouched(self):
        """胎龄列不得被清洗改写(模拟链路要用原始圈序)。"""
        pd.testing.assert_series_equal(self.flagged["TyreLife"],
                                       self.raw["TyreLife"],
                                       check_names=False)

    def test_input_not_mutated(self):
        self.assertNotIn("include_in_fit", self.raw.columns)


class TestChains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = make_laps()
        cls.flagged = lc.add_lap_flags(cls.raw, fuel_effect=lc.FUEL_EFFECT_SEC_PER_LAP)

    def test_fit_ready_subset(self):
        sub = lc.fit_ready(self.flagged)
        self.assertTrue(set(sub["LapNumber"]) <= set(self.flagged["LapNumber"]))
        self.assertEqual(len(sub), int(self.flagged["include_in_fit"].sum()))

    def test_fit_ready_drop_pit_out(self):
        sub = lc.fit_ready(self.flagged, drop_pit_out=True)
        self.assertLessEqual(len(sub), len(lc.fit_ready(self.flagged)))

    def test_sim_ready_keeps_all(self):
        self.assertEqual(len(lc.sim_ready(self.flagged)), len(self.flagged))

    def test_summarise_flags(self):
        info = lc.summarise_flags(self.flagged, "单测")
        self.assertEqual(info["总圈数"], len(self.flagged))
        self.assertEqual(info["参与拟合"],
                         int(self.flagged["include_in_fit"].sum()))
        self.assertLessEqual(info["参与拟合"], info["总圈数"])

    def test_status_columns(self):
        cols = lc.status_export_columns()
        for name in ("LapNumber", "Driver", "Compound", "LapTime_s"):
            self.assertIn(name, cols)


if __name__ == "__main__":
    unittest.main(verbosity=2)
